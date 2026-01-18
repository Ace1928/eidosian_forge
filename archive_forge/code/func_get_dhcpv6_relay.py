from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.dhcp_relay.dhcp_relay import Dhcp_relayArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_dhcpv6_relay(self):
    """Get all DHCPv6 relay configurations available in chassis"""
    dhcpv6_relay_interfaces_path = 'data/openconfig-relay-agent:relay-agent/dhcpv6'
    method = 'GET'
    request = [{'path': dhcpv6_relay_interfaces_path, 'method': method}]
    try:
        response = edit_config(self._module, to_request(self._module, request))
    except ConnectionError as exc:
        self._module.fail_json(msg=str(exc), code=exc.code)
    dhcpv6_relay_interfaces = []
    if response[0][1].get('openconfig-relay-agent:dhcpv6') and response[0][1]['openconfig-relay-agent:dhcpv6'].get('interfaces'):
        dhcpv6_relay_interfaces = response[0][1]['openconfig-relay-agent:dhcpv6']['interfaces'].get('interface', [])
    dhcpv6_relay_configs = {}
    for interface in dhcpv6_relay_interfaces:
        ipv6_dict = {}
        server_addresses = []
        config = interface.get('config', {})
        for address in config.get('helper-address', []):
            temp = {}
            temp['address'] = address
            server_addresses.append(temp)
        ipv6_dict['server_addresses'] = server_addresses
        ipv6_dict['max_hop_count'] = config.get('openconfig-relay-agent-ext:max-hop-count')
        ipv6_dict['source_interface'] = config.get('openconfig-relay-agent-ext:src-intf')
        ipv6_dict['vrf_name'] = config.get('openconfig-relay-agent-ext:vrf')
        opt_config = interface.get('options', {}).get('config', {})
        ipv6_dict['vrf_select'] = opt_config.get('openconfig-relay-agent-ext:vrf-select')
        dhcpv6_relay_configs[interface['id']] = ipv6_dict
    return dhcpv6_relay_configs