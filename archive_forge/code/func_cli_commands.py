from __future__ import absolute_import, division, print_function
import copy
import re
import shlex
import time
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.parsing import Conditional
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from collections import deque
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def cli_commands(self):
    commands = self.normalized_commands
    commands = self.convert_commands_cli(commands)
    if self.chdir:
        for command in commands:
            self.addon_chdir(command)
    if not self.is_tmsh:
        for command in commands:
            self.addon_tmsh_cli(command)
    for command in commands:
        self.merge_command_dict_cli(command)
    result = [x['command'] for x in commands]
    return result