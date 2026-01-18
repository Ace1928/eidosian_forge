from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def is_mlag_global_info_exist(self):
    """whether mlag global attribute info exist"""
    if not self.mlag_global_info:
        return False
    if self.mlag_system_id:
        if self.mlag_priority_id:
            if self.mlag_global_info['lacpMlagSysId'] == self.mlag_system_id and self.mlag_global_info['lacpMlagPriority'] == self.mlag_priority_id:
                return True
        elif self.mlag_global_info['lacpMlagSysId'] == self.mlag_system_id:
            return True
    if self.mlag_priority_id:
        if self.mlag_system_id:
            if self.mlag_global_info['lacpMlagSysId'] == self.mlag_system_id and self.mlag_global_info['lacpMlagPriority'] == self.mlag_priority_id:
                return True
        elif self.mlag_global_info['lacpMlagPriority'] == self.mlag_priority_id:
            return True
    return False