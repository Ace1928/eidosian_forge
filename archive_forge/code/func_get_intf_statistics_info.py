from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, get_nc_config, get_nc_next
def get_intf_statistics_info(self, stat_info, intf_name):
    """Get interface statistics information"""
    if not intf_name:
        return
    if_type = get_interface_type(intf_name)
    if if_type == 'fcoe-port' or if_type == 'nve' or if_type == 'tunnel' or (if_type == 'vbdif') or (if_type == 'vlanif'):
        return
    if stat_info:
        for eles in stat_info:
            if eles.tag in ['receiveByte', 'sendByte', 'rcvUniPacket', 'rcvMutiPacket', 'rcvBroadPacket', 'sendUniPacket', 'sendMutiPacket', 'sendBroadPacket', 'rcvErrorPacket', 'rcvDropPacket', 'sendErrorPacket', 'sendDropPacket']:
                if eles.tag == 'receiveByte':
                    self.result[intf_name]['Inbound octets(bytes)'] = eles.text
                elif eles.tag == 'rcvUniPacket':
                    self.result[intf_name]['Inbound unicast(pkts)'] = eles.text
                elif eles.tag == 'rcvMutiPacket':
                    self.result[intf_name]['Inbound multicast(pkts)'] = eles.text
                elif eles.tag == 'rcvBroadPacket':
                    self.result[intf_name]['Inbound broadcast(pkts)'] = eles.text
                elif eles.tag == 'rcvErrorPacket':
                    self.result[intf_name]['Inbound error(pkts)'] = eles.text
                elif eles.tag == 'rcvDropPacket':
                    self.result[intf_name]['Inbound drop(pkts)'] = eles.text
                elif eles.tag == 'sendByte':
                    self.result[intf_name]['Outbound octets(bytes)'] = eles.text
                elif eles.tag == 'sendUniPacket':
                    self.result[intf_name]['Outbound unicast(pkts)'] = eles.text
                elif eles.tag == 'sendMutiPacket':
                    self.result[intf_name]['Outbound multicast(pkts)'] = eles.text
                elif eles.tag == 'sendBroadPacket':
                    self.result[intf_name]['Outbound broadcast(pkts)'] = eles.text
                elif eles.tag == 'sendErrorPacket':
                    self.result[intf_name]['Outbound error(pkts)'] = eles.text
                elif eles.tag == 'sendDropPacket':
                    self.result[intf_name]['Outbound drop(pkts)'] = eles.text