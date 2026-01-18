from __future__ import absolute_import, division, print_function
import re
from time import sleep
import itertools
from copy import deepcopy
from time import sleep
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import NetworkConfig
from ansible_collections.community.network.plugins.module_utils.network.icx.icx import load_config, get_config
from ansible.module_utils.connection import Connection, ConnectionError, exec_command
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import conditional, remove_default_spec
def extract_list_from_interface(interface):
    if 'ethernet' in interface:
        if 'to' in interface:
            s = re.search('\\d+\\/\\d+/(?P<low>\\d+)\\sto\\s+\\d+\\/\\d+/(?P<high>\\d+)', interface)
            low = int(s.group('low'))
            high = int(s.group('high'))
        else:
            s = re.search('\\d+\\/\\d+/(?P<low>\\d+)', interface)
            low = int(s.group('low'))
            high = int(s.group('low'))
    elif 'lag' in interface:
        if 'to' in interface:
            s = re.search('(?P<low>\\d+)\\sto\\s(?P<high>\\d+)', interface)
            low = int(s.group('low'))
            high = int(s.group('high'))
        else:
            s = re.search('(?P<low>\\d+)', interface)
            low = int(s.group('low'))
            high = int(s.group('low'))
    return (low, high)