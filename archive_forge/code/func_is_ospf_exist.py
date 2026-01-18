from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def is_ospf_exist(self):
    """is ospf exist"""
    if not self.ospf_info:
        return False
    for ospf_site in self.ospf_info['ospfsite']:
        if ospf_site['processId'] == self.ospf:
            return True
        else:
            continue
    return False