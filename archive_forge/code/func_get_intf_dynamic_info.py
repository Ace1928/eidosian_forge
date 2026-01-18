from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, get_nc_config, get_nc_next
def get_intf_dynamic_info(self, dyn_info, intf_name):
    """Get interface dynamic information"""
    if not intf_name:
        return
    if dyn_info:
        for eles in dyn_info:
            if eles.tag in ['ifPhyStatus', 'ifV4State', 'ifV6State', 'ifLinkStatus']:
                if eles.tag == 'ifPhyStatus':
                    self.result[intf_name]['Current physical state'] = eles.text
                elif eles.tag == 'ifLinkStatus':
                    self.result[intf_name]['Current link state'] = eles.text
                elif eles.tag == 'ifV4State':
                    self.result[intf_name]['Current IPv4 state'] = eles.text
                elif eles.tag == 'ifV6State':
                    self.result[intf_name]['Current IPv6 state'] = eles.text