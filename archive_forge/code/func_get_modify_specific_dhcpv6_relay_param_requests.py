from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_modify_specific_dhcpv6_relay_param_requests(self, command):
    """Get requests to modify specific DHCPv6 relay configurations
        based on the command specified for the interface
        """
    requests = []
    name = command['name']
    ipv6 = command.get('ipv6')
    if not ipv6:
        return requests
    server_addresses = self.get_server_addresses(ipv6.get('server_addresses'))
    if server_addresses:
        payload = {'openconfig-relay-agent:helper-address': list(server_addresses)}
        url = self.dhcpv6_relay_intf_config_path['server_addresses_all'].format(intf_name=name)
        requests.append({'path': url, 'method': PATCH, 'data': payload})
    if ipv6.get('vrf_name'):
        payload = {'openconfig-relay-agent-ext:vrf': ipv6['vrf_name']}
        url = self.dhcpv6_relay_intf_config_path['vrf_name'].format(intf_name=name)
        requests.append({'path': url, 'method': PATCH, 'data': payload})
    if ipv6.get('source_interface'):
        payload = {'openconfig-relay-agent-ext:src-intf': ipv6['source_interface']}
        url = self.dhcpv6_relay_intf_config_path['source_interface'].format(intf_name=name)
        requests.append({'path': url, 'method': PATCH, 'data': payload})
    if ipv6.get('max_hop_count'):
        payload = {'openconfig-relay-agent-ext:max-hop-count': ipv6['max_hop_count']}
        url = self.dhcpv6_relay_intf_config_path['max_hop_count'].format(intf_name=name)
        requests.append({'path': url, 'method': PATCH, 'data': payload})
    if ipv6.get('vrf_select') is not None:
        vrf_select = BOOL_TO_SELECT_VALUE[ipv6['vrf_select']]
        payload = {'openconfig-relay-agent-ext:vrf-select': vrf_select}
        url = self.dhcpv6_relay_intf_config_path['vrf_select'].format(intf_name=name)
        requests.append({'path': url, 'method': PATCH, 'data': payload})
    return requests