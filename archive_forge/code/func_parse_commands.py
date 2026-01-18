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
def parse_commands(self):
    results = []
    commands = self._transform_to_complex_commands(self.commands)
    for index, item in enumerate(commands):
        output = item.pop('output', None)
        if output == 'one-line' and 'one-line' not in item['command']:
            item['command'] += ' one-line'
        elif output == 'text' and 'one-line' in item['command']:
            item['command'] = item['command'].replace('one-line', '')
        results.append(item)
    return results