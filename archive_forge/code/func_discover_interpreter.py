from __future__ import (absolute_import, division, print_function)
import bisect
import json
import pkgutil
import re
from ansible import constants as C
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.distro import LinuxDistribution
from ansible.utils.display import Display
from ansible.utils.plugin_docs import get_versioned_doclink
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.facts.system.distribution import Distribution
from traceback import format_exc
def discover_interpreter(action, interpreter_name, discovery_mode, task_vars):
    if interpreter_name != 'python':
        raise ValueError('Interpreter discovery not supported for {0}'.format(interpreter_name))
    host = task_vars.get('inventory_hostname', 'unknown')
    res = None
    platform_type = 'unknown'
    found_interpreters = [u'/usr/bin/python']
    is_auto_legacy = discovery_mode.startswith('auto_legacy')
    is_silent = discovery_mode.endswith('_silent')
    try:
        platform_python_map = C.config.get_config_value('_INTERPRETER_PYTHON_DISTRO_MAP', variables=task_vars)
        bootstrap_python_list = C.config.get_config_value('INTERPRETER_PYTHON_FALLBACK', variables=task_vars)
        display.vvv(msg=u'Attempting {0} interpreter discovery'.format(interpreter_name), host=host)
        command_list = ["command -v '%s'" % py for py in bootstrap_python_list]
        shell_bootstrap = 'echo PLATFORM; uname; echo FOUND; {0}; echo ENDFOUND'.format('; '.join(command_list))
        res = action._low_level_execute_command(shell_bootstrap, sudoable=False)
        raw_stdout = res.get('stdout', u'')
        match = foundre.match(raw_stdout)
        if not match:
            display.debug(u'raw interpreter discovery output: {0}'.format(raw_stdout), host=host)
            raise ValueError('unexpected output from Python interpreter discovery')
        platform_type = match.groups()[0].lower().strip()
        found_interpreters = [interp.strip() for interp in match.groups()[1].splitlines() if interp.startswith('/')]
        display.debug(u'found interpreters: {0}'.format(found_interpreters), host=host)
        if not found_interpreters:
            if not is_silent:
                action._discovery_warnings.append(u'No python interpreters found for host {0} (tried {1})'.format(host, bootstrap_python_list))
            return u'/usr/bin/python'
        if platform_type != 'linux':
            raise NotImplementedError('unsupported platform for extended discovery: {0}'.format(to_native(platform_type)))
        platform_script = pkgutil.get_data('ansible.executor.discovery', 'python_target.py')
        if action._connection.has_pipelining:
            res = action._low_level_execute_command(found_interpreters[0], sudoable=False, in_data=platform_script)
        else:
            raise NotImplementedError('pipelining support required for extended interpreter discovery')
        platform_info = json.loads(res.get('stdout'))
        distro, version = _get_linux_distro(platform_info)
        if not distro or not version:
            raise NotImplementedError('unable to get Linux distribution/version info')
        family = OS_FAMILY_LOWER.get(distro.lower().strip())
        version_map = platform_python_map.get(distro.lower().strip()) or platform_python_map.get(family)
        if not version_map:
            raise NotImplementedError('unsupported Linux distribution: {0}'.format(distro))
        platform_interpreter = to_text(_version_fuzzy_match(version, version_map), errors='surrogate_or_strict')
        if is_auto_legacy:
            if platform_interpreter != u'/usr/bin/python' and u'/usr/bin/python' in found_interpreters:
                if not is_silent:
                    action._discovery_warnings.append(u'Distribution {0} {1} on host {2} should use {3}, but is using /usr/bin/python for backward compatibility with prior Ansible releases. See {4} for more information'.format(distro, version, host, platform_interpreter, get_versioned_doclink('reference_appendices/interpreter_discovery.html')))
                return u'/usr/bin/python'
        if platform_interpreter not in found_interpreters:
            if platform_interpreter not in bootstrap_python_list:
                if not is_silent:
                    action._discovery_warnings.append(u'Platform interpreter {0} on host {1} is missing from bootstrap list'.format(platform_interpreter, host))
            if not is_silent:
                action._discovery_warnings.append(u'Distribution {0} {1} on host {2} should use {3}, but is using {4}, since the discovered platform python interpreter was not present. See {5} for more information.'.format(distro, version, host, platform_interpreter, found_interpreters[0], get_versioned_doclink('reference_appendices/interpreter_discovery.html')))
            return found_interpreters[0]
        return platform_interpreter
    except NotImplementedError as ex:
        display.vvv(msg=u'Python interpreter discovery fallback ({0})'.format(to_text(ex)), host=host)
    except Exception as ex:
        if not is_silent:
            display.warning(msg=u'Unhandled error in Python interpreter discovery for host {0}: {1}'.format(host, to_text(ex)))
            display.debug(msg=u'Interpreter discovery traceback:\n{0}'.format(to_text(format_exc())), host=host)
            if res and res.get('stderr'):
                display.vvv(msg=u'Interpreter discovery remote stderr:\n{0}'.format(to_text(res.get('stderr'))), host=host)
    if not is_silent:
        action._discovery_warnings.append(u'Platform {0} on host {1} is using the discovered Python interpreter at {2}, but future installation of another Python interpreter could change the meaning of that path. See {3} for more information.'.format(platform_type, host, found_interpreters[0], get_versioned_doclink('reference_appendices/interpreter_discovery.html')))
    return found_interpreters[0]