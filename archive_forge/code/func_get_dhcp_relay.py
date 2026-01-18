from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.dhcp_relay.dhcp_relay import Dhcp_relayArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_dhcp_relay(self):
    """Get all DHCP relay configurations available in chassis"""
    dhcp_relay_interfaces_path = 'data/openconfig-relay-agent:relay-agent/dhcp'
    method = 'GET'
    request = [{'path': dhcp_relay_interfaces_path, 'method': method}]
    try:
        response = edit_config(self._module, to_request(self._module, request))
    except ConnectionError as exc:
        self._module.fail_json(msg=str(exc), code=exc.code)
    dhcp_relay_interfaces = []
    if response[0][1].get('openconfig-relay-agent:dhcp') and response[0][1]['openconfig-relay-agent:dhcp'].get('interfaces'):
        dhcp_relay_interfaces = response[0][1]['openconfig-relay-agent:dhcp']['interfaces'].get('interface', [])
    dhcp_relay_configs = {}
    for interface in dhcp_relay_interfaces:
        ipv4_dict = {}
        server_addresses = []
        config = interface.get('config', {})
        for address in config.get('helper-address', []):
            temp = {}
            temp['address'] = address
            server_addresses.append(temp)
        ipv4_dict['server_addresses'] = server_addresses
        ipv4_dict['max_hop_count'] = config.get('openconfig-relay-agent-ext:max-hop-count')
        ipv4_dict['policy_action'] = config.get('openconfig-relay-agent-ext:policy-action')
        ipv4_dict['source_interface'] = config.get('openconfig-relay-agent-ext:src-intf')
        ipv4_dict['vrf_name'] = config.get('openconfig-relay-agent-ext:vrf')
        opt_config = interface.get('agent-information-option', {}).get('config', {})
        ipv4_dict['circuit_id'] = opt_config.get('circuit-id')
        ipv4_dict['link_select'] = opt_config.get('openconfig-relay-agent-ext:link-select')
        ipv4_dict['vrf_select'] = opt_config.get('openconfig-relay-agent-ext:vrf-select')
        dhcp_relay_configs[interface['id']] = ipv4_dict
    return dhcp_relay_configs