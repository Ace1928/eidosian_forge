from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def data_init(self):
    """data init"""
    if self.l2_sub_interface:
        self.l2_sub_interface = self.l2_sub_interface.replace(' ', '').upper()
    if self.encapsulation and self.l2_sub_interface:
        self.l2sub_info = self.get_l2_sub_intf_dict(self.l2_sub_interface)
    if self.bridge_domain_id:
        if self.bind_vlan_id or self.l2_sub_interface:
            self.vap_info = self.get_bd_vap_dict()