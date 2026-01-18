from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, get_nc_config, get_nc_next
def get_link_status(self):
    """Get link status information"""
    if self.param_type == INTERFACE_FULL_NAME:
        self.init_interface_data(self.interface)
        self.get_interface_info()
        if is_ethernet_port(self.interface):
            self.get_port_info(self.interface)
    elif self.param_type == INTERFACE_TYPE:
        self.get_all_interface_info(self.interface)
    else:
        self.get_all_interface_info()