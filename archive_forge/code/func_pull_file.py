from __future__ import absolute_import, division, print_function
import json
import re
from ansible.errors import AnsibleConnectionFailure
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.common._collections_compat import Mapping
from ansible.module_utils.connection import ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.ansible.netcommon.plugins.plugin_utils.cliconf_base import CliconfBase
def pull_file(self, command, remotepassword=None):
    possible_errors_re = [re.compile(b'timed out'), re.compile(b'(?i)No space.*#'), re.compile(b'(?i)Permission denied.*#'), re.compile(b'(?i)No such file.*#'), re.compile(b'Compaction is not supported on this platform.*#'), re.compile(b'Compact of.*failed.*#'), re.compile(b'(?i)Could not resolve hostname'), re.compile(b'(?i)Too many authentication failures'), re.compile(b'Access Denied'), re.compile(b'(?i)Copying to\\/from this server name is not permitted')]
    current_stderr_re = self._connection._get_terminal_std_re('terminal_stderr_re')
    current_stderr_re.extend(possible_errors_re)
    possible_prompts_re = [re.compile(b'file existing with this name'), re.compile(b'sure you want to continue connecting'), re.compile(b'(?i)Password:.*')]
    current_stdout_re = self._connection._get_terminal_std_re('terminal_stdout_re')
    current_stdout_re.extend(possible_prompts_re)
    retry = 1
    file_pulled = False
    try:
        while not file_pulled and retry <= 6:
            retry += 1
            output = self.send_command(command=command, strip_prompt=False)
            if possible_prompts_re[0].search(to_bytes(output)):
                output = self.send_command(command='y', strip_prompt=False)
            if possible_prompts_re[1].search(to_bytes(output)):
                output = self.send_command(command='yes', strip_prompt=False)
            if possible_prompts_re[2].search(to_bytes(output)):
                output = self.send_command(command=remotepassword, strip_prompt=False)
            if 'Copy complete' in output:
                file_pulled = True
        return file_pulled
    finally:
        for x in possible_prompts_re:
            current_stdout_re.remove(x)
        for x in possible_errors_re:
            current_stderr_re.remove(x)