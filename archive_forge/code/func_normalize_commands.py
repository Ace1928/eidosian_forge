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
@staticmethod
def normalize_commands(raw_commands):
    if not raw_commands:
        return None
    result = []
    for command in raw_commands:
        command = command.strip()
        if command[0:5] == 'tmsh ':
            command = command[4:].strip()
        result.append(command)
    return result