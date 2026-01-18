from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, get_nc_config, get_nc_next
def get_intf_cleared_stat(self, clr_stat, intf_name):
    """Get interface cleared state information"""
    if not intf_name:
        return
    if_type = get_interface_type(intf_name)
    if if_type == 'fcoe-port' or if_type == 'nve' or if_type == 'tunnel' or (if_type == 'vbdif') or (if_type == 'vlanif'):
        return
    if clr_stat:
        for eles in clr_stat:
            if eles.tag in ['inByteRate', 'inPacketRate', 'outByteRate', 'outPacketRate']:
                if eles.tag == 'inByteRate':
                    self.result[intf_name]['Inbound rate(byte/sec)'] = eles.text
                elif eles.tag == 'inPacketRate':
                    self.result[intf_name]['Inbound rate(pkts/sec)'] = eles.text
                elif eles.tag == 'outByteRate':
                    self.result[intf_name]['Outbound rate(byte/sec)'] = eles.text
                elif eles.tag == 'outPacketRate':
                    self.result[intf_name]['Outbound rate(pkts/sec)'] = eles.text