from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, get_nc_config, set_nc_config
def get_interface_vpn(self):
    """ get the VPN instance associated with the interface"""
    xml_str = CE_NC_GET_VRF_INTERFACE
    con_obj = get_nc_config(self.module, xml_str)
    if '<data/>' in con_obj:
        return
    xml_str = con_obj.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
    root = ElementTree.fromstring(xml_str)
    vpns = root.findall('l3vpn/l3vpncomm/l3vpnInstances/l3vpnInstance')
    if vpns:
        for vpnele in vpns:
            vpn_name = None
            for vpninfo in vpnele:
                if vpninfo.tag == 'vrfName':
                    vpn_name = vpninfo.text
                if vpninfo.tag == 'l3vpnIfs':
                    self.get_interface_vpn_name(vpninfo, vpn_name)
    return