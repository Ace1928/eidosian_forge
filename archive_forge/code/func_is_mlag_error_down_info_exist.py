from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def is_mlag_error_down_info_exist(self):
    """whether mlag error down info exist"""
    if not self.mlag_error_down_info:
        return False
    for info in self.mlag_error_down_info['mlagErrorDownInfos']:
        if info['portName'].upper() == self.interface.upper():
            return True
    return False