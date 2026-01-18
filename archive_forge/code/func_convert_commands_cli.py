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
def convert_commands_cli(self, commands):
    result = []
    for command in commands:
        tmp = dict(command='', pipeline='')
        pipeline = command.split('|', 1) if self.cmd_has_pipe(command) else [command]
        tmp['command'] = pipeline[0]
        try:
            tmp['pipeline'] = pipeline[1]
        except IndexError:
            pass
        result.append(tmp)
    return result