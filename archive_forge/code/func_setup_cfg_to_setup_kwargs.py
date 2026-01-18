import logging  # noqa
from collections import defaultdict
import io
import os
import re
import shlex
import sys
import traceback
import distutils.ccompiler
from distutils import errors
from distutils import log
import pkg_resources
from setuptools import dist as st_dist
from setuptools import extension
from pbr import extra_files
import pbr.hooks
def setup_cfg_to_setup_kwargs(config, script_args=()):
    """Convert config options to kwargs.

    Processes the setup.cfg options and converts them to arguments accepted
    by setuptools' setup() function.
    """
    kwargs = {}
    all_requirements = {}
    for alias, arg in CFG_TO_PY_SETUP_ARGS:
        section, option = alias
        in_cfg_value = has_get_option(config, section, option)
        if not in_cfg_value and arg == 'long_description':
            in_cfg_value = has_get_option(config, section, 'description_file')
            if in_cfg_value:
                in_cfg_value = split_multiline(in_cfg_value)
                value = ''
                for filename in in_cfg_value:
                    description_file = io.open(filename, encoding='utf-8')
                    try:
                        value += description_file.read().strip() + '\n\n'
                    finally:
                        description_file.close()
                in_cfg_value = value
        if not in_cfg_value:
            continue
        if arg in CSV_FIELDS:
            in_cfg_value = split_csv(in_cfg_value)
        if arg in MULTI_FIELDS:
            in_cfg_value = split_multiline(in_cfg_value)
        elif arg in MAP_FIELDS:
            in_cfg_map = {}
            for i in split_multiline(in_cfg_value):
                k, v = i.split('=', 1)
                in_cfg_map[k.strip()] = v.strip()
            in_cfg_value = in_cfg_map
        elif arg in BOOL_FIELDS:
            if in_cfg_value.lower() in ('true', 't', '1', 'yes', 'y'):
                in_cfg_value = True
            else:
                in_cfg_value = False
        if in_cfg_value:
            if arg in ('install_requires', 'tests_require'):
                in_cfg_value = [_VERSION_SPEC_RE.sub('\\1\\2', pred) for pred in in_cfg_value]
            if arg == 'install_requires':
                install_requires = []
                requirement_pattern = '(?P<package>[^;]*);?(?P<env_marker>[^#]*?)(?:\\s*#.*)?$'
                for requirement in in_cfg_value:
                    m = re.match(requirement_pattern, requirement)
                    requirement_package = m.group('package').strip()
                    env_marker = m.group('env_marker').strip()
                    install_requires.append((requirement_package, env_marker))
                all_requirements[''] = install_requires
            elif arg == 'package_dir':
                in_cfg_value = {'': in_cfg_value}
            elif arg in ('package_data', 'data_files'):
                data_files = {}
                firstline = True
                prev = None
                for line in in_cfg_value:
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key_unquoted = shlex_split(key.strip())[0]
                        key, value = (key_unquoted, value.strip())
                        if key in data_files:
                            prev = data_files[key]
                            prev.extend(shlex_split(value))
                        else:
                            prev = data_files[key.strip()] = shlex_split(value)
                    elif firstline:
                        raise errors.DistutilsOptionError('malformed package_data first line %r (misses "=")' % line)
                    else:
                        prev.extend(shlex_split(line.strip()))
                    firstline = False
                if arg == 'data_files':
                    data_files = sorted(data_files.items())
                in_cfg_value = data_files
            elif arg == 'cmdclass':
                cmdclass = {}
                dist = st_dist.Distribution()
                for cls_name in in_cfg_value:
                    cls = resolve_name(cls_name)
                    cmd = cls(dist)
                    cmdclass[cmd.get_command_name()] = cls
                in_cfg_value = cmdclass
        kwargs[arg] = in_cfg_value
    if 'extras' in config:
        requirement_pattern = '(?P<package>[^:]*):?(?P<env_marker>[^#]*?)(?:\\s*#.*)?$'
        extras = config['extras']
        if 'test' not in extras:
            from pbr import packaging
            extras['test'] = '\n'.join(packaging.parse_requirements(packaging.TEST_REQUIREMENTS_FILES)).replace(';', ':')
        for extra in extras:
            extra_requirements = []
            requirements = split_multiline(extras[extra])
            for requirement in requirements:
                m = re.match(requirement_pattern, requirement)
                extras_value = m.group('package').strip()
                env_marker = m.group('env_marker')
                extra_requirements.append((extras_value, env_marker))
            all_requirements[extra] = extra_requirements
    extras_require = {}
    for req_group in all_requirements:
        for requirement, env_marker in all_requirements[req_group]:
            if env_marker:
                extras_key = '%s:(%s)' % (req_group, env_marker)
                if 'bdist_wheel' not in script_args:
                    try:
                        if pkg_resources.evaluate_marker('(%s)' % env_marker):
                            extras_key = req_group
                    except SyntaxError:
                        log.error('Marker evaluation failed, see the following error.  For more information see: http://docs.openstack.org/pbr/latest/user/using.html#environment-markers')
                        raise
            else:
                extras_key = req_group
            extras_require.setdefault(extras_key, []).append(requirement)
    kwargs['install_requires'] = extras_require.pop('', [])
    kwargs['extras_require'] = extras_require
    return kwargs