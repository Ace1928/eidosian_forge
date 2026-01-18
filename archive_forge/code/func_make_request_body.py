from __future__ import absolute_import, division, print_function
import re
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
def make_request_body(self):
    iface = self.get_target_interface()
    body = {'iscsiInterface': iface['ioInterfaceTypeData'][iface['ioInterfaceTypeData']['interfaceType']]['id'], 'settings': {'tcpListenPort': [], 'ipv4Address': [self.address], 'ipv4SubnetMask': [], 'ipv4GatewayAddress': [], 'ipv4AddressConfigMethod': [], 'maximumFramePayloadSize': [], 'ipv4VlanId': [], 'ipv4OutboundPacketPriority': [], 'ipv4Enabled': [], 'ipv6Enabled': [], 'ipv6LocalAddresses': [], 'ipv6RoutableAddresses': [], 'ipv6PortRouterAddress': [], 'ipv6AddressConfigMethod': [], 'ipv6OutboundPacketPriority': [], 'ipv6VlanId': [], 'ipv6HopLimit': [], 'ipv6NdReachableTime': [], 'ipv6NdRetransmitTime': [], 'ipv6NdStaleTimeout': [], 'ipv6DuplicateAddressDetectionAttempts': [], 'maximumInterfaceSpeed': []}}
    return body