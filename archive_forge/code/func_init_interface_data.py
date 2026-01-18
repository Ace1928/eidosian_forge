from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, get_nc_config, get_nc_next
def init_interface_data(self, intf_name):
    """Init interface data"""
    self.result[intf_name] = dict()
    self.result[intf_name]['Current physical state'] = 'down'
    self.result[intf_name]['Current link state'] = 'down'
    self.result[intf_name]['Current IPv4 state'] = 'down'
    self.result[intf_name]['Current IPv6 state'] = 'down'
    self.result[intf_name]['Inbound octets(bytes)'] = '--'
    self.result[intf_name]['Inbound unicast(pkts)'] = '--'
    self.result[intf_name]['Inbound multicast(pkts)'] = '--'
    self.result[intf_name]['Inbound broadcast(pkts)'] = '--'
    self.result[intf_name]['Inbound error(pkts)'] = '--'
    self.result[intf_name]['Inbound drop(pkts)'] = '--'
    self.result[intf_name]['Inbound rate(byte/sec)'] = '--'
    self.result[intf_name]['Inbound rate(pkts/sec)'] = '--'
    self.result[intf_name]['Outbound octets(bytes)'] = '--'
    self.result[intf_name]['Outbound unicast(pkts)'] = '--'
    self.result[intf_name]['Outbound multicast(pkts)'] = '--'
    self.result[intf_name]['Outbound broadcast(pkts)'] = '--'
    self.result[intf_name]['Outbound error(pkts)'] = '--'
    self.result[intf_name]['Outbound drop(pkts)'] = '--'
    self.result[intf_name]['Outbound rate(byte/sec)'] = '--'
    self.result[intf_name]['Outbound rate(pkts/sec)'] = '--'
    self.result[intf_name]['Speed'] = '--'