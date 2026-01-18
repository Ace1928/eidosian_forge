from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def is_valid_vrf_name(self):
    """check whether the input ospf vrf name is valid"""
    if len(self.vrf) > 31 or len(self.vrf) < 1:
        return False
    if self.vrf.find('?') != -1:
        return False
    if self.vrf.find(' ') != -1:
        return False
    return True