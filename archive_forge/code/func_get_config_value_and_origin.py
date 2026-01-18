from __future__ import (absolute_import, division, print_function)
import atexit
import configparser
import os
import os.path
import sys
import stat
import tempfile
from collections import namedtuple
from collections.abc import Mapping, Sequence
from jinja2.nativetypes import NativeEnvironment
from ansible.errors import AnsibleOptionsError, AnsibleError
from ansible.module_utils.common.text.converters import to_text, to_bytes, to_native
from ansible.module_utils.common.yaml import yaml_load
from ansible.module_utils.six import string_types
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.parsing.quoting import unquote
from ansible.parsing.yaml.objects import AnsibleVaultEncryptedUnicode
from ansible.utils import py3compat
from ansible.utils.path import cleanup_tmp_file, makedirs_safe, unfrackpath
def get_config_value_and_origin(self, config, cfile=None, plugin_type=None, plugin_name=None, keys=None, variables=None, direct=None):
    """ Given a config key figure out the actual value and report on the origin of the settings """
    if cfile is None:
        cfile = self._config_file
    if config == 'CONFIG_FILE':
        return (cfile, '')
    value = None
    origin = None
    defs = self.get_configuration_definitions(plugin_type, plugin_name)
    if config in defs:
        aliases = defs[config].get('aliases', [])
        if direct:
            if config in direct:
                value = direct[config]
                origin = 'Direct'
            else:
                direct_aliases = [direct[alias] for alias in aliases if alias in direct]
                if direct_aliases:
                    value = direct_aliases[0]
                    origin = 'Direct'
        if value is None and variables and defs[config].get('vars'):
            value, origin = self._loop_entries(variables, defs[config]['vars'])
            origin = 'var: %s' % origin
        if value is None and defs[config].get('keyword') and keys:
            value, origin = self._loop_entries(keys, defs[config]['keyword'])
            origin = 'keyword: %s' % origin
        if value is None and keys:
            if config in keys:
                value = keys[config]
                keyword = config
            elif aliases:
                for alias in aliases:
                    if alias in keys:
                        value = keys[alias]
                        keyword = alias
                        break
            if value is not None:
                origin = 'keyword: %s' % keyword
        if value is None and 'cli' in defs[config]:
            from ansible import context
            value, origin = self._loop_entries(context.CLIARGS, defs[config]['cli'])
            origin = 'cli: %s' % origin
        if value is None and defs[config].get('env'):
            value, origin = self._loop_entries(py3compat.environ, defs[config]['env'])
            origin = 'env: %s' % origin
        if self._parsers.get(cfile, None) is None:
            self._parse_config_file(cfile)
        if value is None and cfile is not None:
            ftype = get_config_type(cfile)
            if ftype and defs[config].get(ftype):
                if ftype == 'ini':
                    try:
                        for ini_entry in defs[config]['ini']:
                            temp_value = get_ini_config_value(self._parsers[cfile], ini_entry)
                            if temp_value is not None:
                                value = temp_value
                                origin = cfile
                                if 'deprecated' in ini_entry:
                                    self.DEPRECATED.append(('[%s]%s' % (ini_entry['section'], ini_entry['key']), ini_entry['deprecated']))
                    except Exception as e:
                        sys.stderr.write('Error while loading ini config %s: %s' % (cfile, to_native(e)))
                elif ftype == 'yaml':
                    origin = cfile
        if value is None:
            if defs[config].get('required', False):
                if not plugin_type or config not in INTERNAL_DEFS.get(plugin_type, {}):
                    raise AnsibleError('No setting was provided for required configuration %s' % to_native(_get_entry(plugin_type, plugin_name, config)))
            else:
                origin = 'default'
                value = self.template_default(defs[config].get('default'), variables)
        try:
            value = ensure_type(value, defs[config].get('type'), origin=origin)
        except ValueError as e:
            if origin.startswith('env:') and value == '':
                origin = 'default'
                value = ensure_type(defs[config].get('default'), defs[config].get('type'), origin=origin)
            else:
                raise AnsibleOptionsError('Invalid type for configuration option %s (from %s): %s' % (to_native(_get_entry(plugin_type, plugin_name, config)).strip(), origin, to_native(e)))
        if value is not None and 'choices' in defs[config] and (defs[config]['choices'] is not None):
            invalid_choices = True
            if defs[config].get('type') == 'list':
                invalid_choices = not all((choice in defs[config]['choices'] for choice in value))
            else:
                invalid_choices = value not in defs[config]['choices']
            if invalid_choices:
                if isinstance(defs[config]['choices'], Mapping):
                    valid = ', '.join([to_text(k) for k in defs[config]['choices'].keys()])
                elif isinstance(defs[config]['choices'], string_types):
                    valid = defs[config]['choices']
                elif isinstance(defs[config]['choices'], Sequence):
                    valid = ', '.join([to_text(c) for c in defs[config]['choices']])
                else:
                    valid = defs[config]['choices']
                raise AnsibleOptionsError('Invalid value "%s" for configuration option "%s", valid values are: %s' % (value, to_native(_get_entry(plugin_type, plugin_name, config)), valid))
        if 'deprecated' in defs[config] and origin != 'default':
            self.DEPRECATED.append((config, defs[config].get('deprecated')))
    else:
        raise AnsibleError('Requested entry (%s) was not defined in configuration.' % to_native(_get_entry(plugin_type, plugin_name, config)))
    return (value, origin)