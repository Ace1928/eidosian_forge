from __future__ import (absolute_import, division, print_function)
from ansible.cli import CLI
import os
import yaml
import shlex
import subprocess
from collections.abc import Mapping
from ansible import context
import ansible.plugins.loader as plugin_loader
from ansible import constants as C
from ansible.cli.arguments import option_helpers as opt_help
from ansible.config.manager import ConfigManager, Setting
from ansible.errors import AnsibleError, AnsibleOptionsError
from ansible.module_utils.common.text.converters import to_native, to_text, to_bytes
from ansible.module_utils.common.json import json_dump
from ansible.module_utils.six import string_types
from ansible.parsing.quoting import is_quoted
from ansible.parsing.yaml.dumper import AnsibleDumper
from ansible.utils.color import stringc
from ansible.utils.display import Display
from ansible.utils.path import unfrackpath
def execute_dump(self):
    """
        Shows the current settings, merges ansible.cfg if specified
        """
    if context.CLIARGS['type'] == 'base':
        output = self._get_global_configs()
    elif context.CLIARGS['type'] == 'all':
        output = self._get_global_configs()
        for ptype in C.CONFIGURABLE_PLUGINS:
            plugin_list = self._get_plugin_configs(ptype, context.CLIARGS['args'])
            if context.CLIARGS['format'] == 'display':
                if not context.CLIARGS['only_changed'] or plugin_list:
                    output.append('\n%s:\n%s' % (ptype.upper(), '=' * len(ptype)))
                    output.extend(plugin_list)
            else:
                if ptype in ('modules', 'doc_fragments'):
                    pname = ptype.upper()
                else:
                    pname = '%s_PLUGINS' % ptype.upper()
                output.append({pname: plugin_list})
    else:
        output = self._get_plugin_configs(context.CLIARGS['type'], context.CLIARGS['args'])
    if context.CLIARGS['format'] == 'display':
        text = '\n'.join(output)
    if context.CLIARGS['format'] == 'yaml':
        text = yaml_dump(output)
    elif context.CLIARGS['format'] == 'json':
        text = json_dump(output)
    self.pager(to_text(text, errors='surrogate_or_strict'))