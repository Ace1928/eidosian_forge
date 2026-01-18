from __future__ import absolute_import, division, print_function
import json
import re
from ansible.errors import AnsibleConnectionFailure
from ansible.module_utils._text import to_bytes, to_text
from ansible_collections.ansible.netcommon.plugins.plugin_utils.terminal_base import TerminalBase
def disable_pager(self):
    try:
        self._exec_cli_command('no terminal pager')
    except AnsibleConnectionFailure:
        raise AnsibleConnectionFailure('unable to disable terminal pager')