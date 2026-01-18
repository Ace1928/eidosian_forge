from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def is_valid_bandwidth(self):
    """check whether the input ospf bandwidth reference is valid"""
    if not self.bandwidth.isdigit():
        return False
    if int(self.bandwidth) > 2147483648 or int(self.bandwidth) < 1:
        return False
    return True