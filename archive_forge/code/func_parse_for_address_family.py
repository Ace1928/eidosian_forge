from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.ospfv3.ospfv3 import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.ospfv3 import (
def parse_for_address_family(self, current):
    """Parsing and Fishing out address family contents"""
    pid_addr_family_dict = {}
    temp_dict = {}
    temp_pid = None
    temp = []
    if current.get('address_family'):
        for each in current.pop('address_family'):
            each = utils.remove_empties(each)
            if each.get('exit'):
                if temp_pid == each.get('exit')['pid']:
                    temp.append(temp_dict)
                    pid_addr_family_dict[temp_pid] = temp
                    temp_dict = dict()
                else:
                    temp_pid = each.get('exit')['pid']
                    pid_addr_family_dict[temp_pid] = [temp_dict]
                    temp = []
                    temp.append(temp_dict)
                    temp_dict = dict()
            elif each.get('manet') and temp_dict.get('manet'):
                for k, v in iteritems(each.get('manet')):
                    if k in temp_dict.get('manet'):
                        temp_dict.get('manet')[k].update(v)
                    else:
                        temp_dict['manet'].update(each.get('manet'))
            elif each.get('manet') and (not temp_dict.get('manet')):
                temp_dict['manet'] = each.get('manet')
            else:
                temp_dict.update(each)
    return pid_addr_family_dict