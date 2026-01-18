from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_delete_specific_dhcp_relay_param_requests(self, command, config, is_state_deleted=True):
    """Get requests to delete specific DHCP relay configurations
        based on the command specified for the interface
        """
    requests = []
    name = command['name']
    ipv4 = command.get('ipv4')
    have_ipv4 = config.get('ipv4')
    if not ipv4 or not have_ipv4:
        return requests
    server_addresses = self.get_server_addresses(ipv4.get('server_addresses'))
    have_server_addresses = self.get_server_addresses(have_ipv4.get('server_addresses'))
    if ipv4.get('server_addresses') and len(ipv4.get('server_addresses')) and (not server_addresses):
        requests.append(self.get_delete_all_dhcp_relay_intf_request(name))
        return requests
    del_server_addresses = have_server_addresses.intersection(server_addresses)
    if del_server_addresses:
        if is_state_deleted and len(del_server_addresses) == len(have_server_addresses):
            requests.append(self.get_delete_all_dhcp_relay_intf_request(name))
            return requests
        for addr in del_server_addresses:
            url = self.dhcp_relay_intf_config_path['server_address'].format(intf_name=name, server_address=addr)
            requests.append({'path': url, 'method': DELETE})
    if ipv4.get('link_select') is not None and have_ipv4.get('link_select'):
        url = self.dhcp_relay_intf_config_path['link_select'].format(intf_name=name)
        requests.append({'path': url, 'method': DELETE})
    if ipv4.get('source_interface') and have_ipv4.get('source_interface') and (ipv4['source_interface'] == have_ipv4['source_interface']):
        url = self.dhcp_relay_intf_config_path['source_interface'].format(intf_name=name)
        requests.append({'path': url, 'method': DELETE})
    if ipv4.get('max_hop_count') and have_ipv4.get('max_hop_count') and (ipv4['max_hop_count'] == have_ipv4['max_hop_count']) and (have_ipv4['max_hop_count'] != DEFAULT_MAX_HOP_COUNT):
        url = self.dhcp_relay_intf_config_path['max_hop_count'].format(intf_name=name)
        requests.append({'path': url, 'method': DELETE})
    if ipv4.get('vrf_select') is not None and have_ipv4.get('vrf_select'):
        url = self.dhcp_relay_intf_config_path['vrf_select'].format(intf_name=name)
        requests.append({'path': url, 'method': DELETE})
    if ipv4.get('policy_action') and have_ipv4.get('policy_action') and (ipv4['policy_action'] == have_ipv4['policy_action']) and (have_ipv4['policy_action'] != DEFAULT_POLICY_ACTION):
        url = self.dhcp_relay_intf_config_path['policy_action'].format(intf_name=name)
        requests.append({'path': url, 'method': DELETE})
    if ipv4.get('circuit_id') and have_ipv4.get('circuit_id') and (ipv4['circuit_id'] == have_ipv4['circuit_id']) and (have_ipv4['circuit_id'] != DEFAULT_CIRCUIT_ID):
        url = self.dhcp_relay_intf_config_path['circuit_id'].format(intf_name=name)
        requests.append({'path': url, 'method': DELETE})
    return requests