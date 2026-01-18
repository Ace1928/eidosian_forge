from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, get_nc_config, set_nc_config
def get_intf_conf_info(self):
    """ get related configuration of the interface"""
    conf_str = CE_NC_GET_INTF % self.vpn_interface
    con_obj = get_nc_config(self.module, conf_str)
    if '<data/>' in con_obj:
        return
    xml_str = con_obj.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
    root = ElementTree.fromstring(xml_str)
    interface = root.find('ifm/interfaces/interface')
    if interface:
        for eles in interface:
            if eles.tag in ['isL2SwitchPort']:
                self.intf_info[eles.tag] = eles.text
    self.get_interface_vpn()
    return