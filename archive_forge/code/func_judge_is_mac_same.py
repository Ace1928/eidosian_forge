from __future__ import (absolute_import, division, print_function)
import copy
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, set_nc_config, get_nc_config, execute_nc_action
def judge_is_mac_same(mac1, mac2):
    """Judge whether two macs are the same"""
    if mac1 == mac2:
        return True
    list1 = re.findall('([0-9A-Fa-f]+)', mac1)
    list2 = re.findall('([0-9A-Fa-f]+)', mac2)
    if len(list1) != len(list2):
        return False
    for index, value in enumerate(list1, start=0):
        if value.lstrip('0').lower() != list2[index].lstrip('0').lower():
            return False
    return True