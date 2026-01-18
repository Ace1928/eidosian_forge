from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.connection import ConnectionError, exec_command
from ansible_collections.community.network.plugins.module_utils.network.icx.icx import run_commands, get_config, load_config
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import CustomNetworkConfig
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import remove_default_spec
def range_to_members(ranges, prefix=''):
    match = re.findall('(ethe[a-z]* [0-9]/[0-9]/[0-9]+)( to [0-9]/[0-9]/[0-9]+)?', ranges)
    members = list()
    for m in match:
        start, end = m
        if end == '':
            start = start.replace('ethe ', 'ethernet ')
            members.append('%s%s' % (prefix, start))
        else:
            start_tmp = re.search('[0-9]/[0-9]/([0-9]+)', start)
            end_tmp = re.search('[0-9]/[0-9]/([0-9]+)', end)
            start = int(start_tmp.group(1))
            end = int(end_tmp.group(1)) + 1
            for num in range(start, end):
                members.append('%sethernet 1/1/%s' % (prefix, num))
    return members