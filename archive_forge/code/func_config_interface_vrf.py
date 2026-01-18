from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, get_nc_config, set_nc_config
def config_interface_vrf(self):
    """ configure VPN instance of the interface"""
    if not self.conf_exist and self.state == 'present':
        xml_str = CE_NC_MERGE_VRF_INTERFACE % (self.vrf, self.vpn_interface)
        ret_xml = set_nc_config(self.module, xml_str)
        self.check_response(ret_xml, 'VRF_INTERFACE_CONFIG')
        self.changed = True
    elif self.state == 'absent':
        xml_str = CE_NC_DEL_INTF_VPN % (self.vrf, self.vpn_interface)
        ret_xml = set_nc_config(self.module, xml_str)
        self.check_response(ret_xml, 'DEL_VRF_INTERFACE_CONFIG')
        self.changed = True