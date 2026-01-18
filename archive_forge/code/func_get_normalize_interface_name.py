from __future__ import absolute_import, division, print_function
import re
import json
import ast
from copy import copy
from itertools import (count, groupby)
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible.module_utils.common.network import (
from ansible.module_utils.common.validation import check_required_arguments
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_normalize_interface_name(intf_name, module):
    change_flag = False
    ret_intf_name = re.sub('\\s+', '', intf_name, flags=re.UNICODE)
    ret_intf_name = ret_intf_name.capitalize()
    match = re.search('\\d', ret_intf_name)
    if match:
        change_flag = True
        start_pos = match.start()
        name = ret_intf_name[0:start_pos]
        intf_id = ret_intf_name[start_pos:]
        if name.startswith('Eth'):
            validate_intf_naming_mode(intf_name, module)
        if ret_intf_name.startswith('Management') or ret_intf_name.startswith('Mgmt'):
            name = 'eth'
            intf_id = '0'
        elif re.search(STANDARD_ETH_REGEXP, ret_intf_name):
            name = 'Eth'
        elif re.search(NATIVE_ETH_REGEXP, ret_intf_name):
            name = 'Ethernet'
        elif name.startswith('Po'):
            name = 'PortChannel'
        elif name.startswith('Vlan'):
            name = 'Vlan'
        elif name.startswith('Lo'):
            name = 'Loopback'
        else:
            change_flag = False
        ret_intf_name = name + intf_id
    if not change_flag:
        ret_intf_name = intf_name
    return ret_intf_name