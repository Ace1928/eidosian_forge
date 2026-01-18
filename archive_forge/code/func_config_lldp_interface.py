from __future__ import (absolute_import, division, print_function)
import copy
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import set_nc_config, get_nc_config
def config_lldp_interface(self):
    """config lldp interface"""
    if self.lldpenable:
        self.config_global_lldp_enable()
    if self.function_lldp_interface_flag == 'disableINTERFACE':
        self.config_interface_lldp_disable_config()
    elif self.function_lldp_interface_flag == 'tlvdisableINTERFACE':
        self.config_interface_tlv_disable_config()
    elif self.function_lldp_interface_flag == 'tlvenableINTERFACE':
        self.config_interface_tlv_enable_config()
    elif self.function_lldp_interface_flag == 'intervalINTERFACE':
        self.config_interface_interval_config()