from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
def is_valid_v6addr(addr):
    """check if ipv6 addr is valid"""
    if addr.find(':') != -1:
        addr_list = addr.split(':')
        if len(addr_list) > 6:
            return False
        if addr_list[1] == '':
            return False
        return True
    return False