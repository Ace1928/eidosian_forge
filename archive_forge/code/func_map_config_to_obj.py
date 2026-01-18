from __future__ import absolute_import, division, print_function
import re
import time
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.validation import check_required_one_of
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.vyos import (
def map_config_to_obj(module):
    objs = []
    output = run_commands(module, 'show interfaces')
    lines = output[0].strip().splitlines()[3:]
    for line in lines:
        splitted_line = re.split('\\s{2,}', line.strip())
        obj = {}
        eth = splitted_line[0].strip("'")
        if eth.startswith('eth'):
            obj['interfaces'] = []
            if '.' in eth:
                interface = eth.split('.')[0]
                obj['interfaces'].append(interface)
                obj['vlan_id'] = eth.split('.')[-1]
            else:
                obj['interfaces'].append(eth)
                obj['vlan_id'] = None
            if splitted_line[1].strip("'") != '-':
                obj['address'] = splitted_line[1].strip("'")
            if len(splitted_line) > 3:
                obj['name'] = splitted_line[3].strip("'")
            obj['state'] = 'present'
            objs.append(obj)
    return objs