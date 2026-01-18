from __future__ import (absolute_import, division, print_function)
import os
import tempfile
import json
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.dellemc_idrac import iDRACConnection, idrac_auth_params
from ansible.module_utils.basic import AnsibleModule
def run_idrac_network_config(idrac, module):
    idrac.use_redfish = True
    share_path = tempfile.gettempdir() + os.sep
    upd_share = file_share_manager.create_share_obj(share_path=share_path, isFolder=True)
    if not upd_share.IsValid:
        module.fail_json(msg='Unable to access the share. Ensure that the share name, share mount, and share credentials provided are correct.')
    idrac.config_mgr.set_liason_share(upd_share)
    if module.params['register_idrac_on_dns'] is not None:
        idrac.config_mgr.configure_dns(register_idrac_on_dns=DNSRegister_NICTypes[module.params['register_idrac_on_dns']])
    if module.params['dns_idrac_name'] is not None:
        idrac.config_mgr.configure_dns(dns_idrac_name=module.params['dns_idrac_name'])
    if module.params['auto_config'] is not None:
        idrac.config_mgr.configure_dns(auto_config=DNSDomainFromDHCP_NICStaticTypes[module.params['auto_config']])
    if module.params['static_dns'] is not None:
        idrac.config_mgr.configure_dns(static_dns=module.params['static_dns'])
    if module.params['setup_idrac_nic_vlan'] is not None:
        idrac.config_mgr.configure_nic_vlan(vlan_enable=VLanEnable_NICTypes[module.params['setup_idrac_nic_vlan']])
    if module.params['vlan_id'] is not None:
        idrac.config_mgr.configure_nic_vlan(vlan_id=module.params['vlan_id'])
    if module.params['vlan_priority'] is not None:
        idrac.config_mgr.configure_nic_vlan(vlan_priority=module.params['vlan_priority'])
    if module.params['enable_nic'] is not None:
        idrac.config_mgr.configure_network_settings(enable_nic=Enable_NICTypes[module.params['enable_nic']])
    if module.params['nic_selection'] is not None:
        idrac.config_mgr.configure_network_settings(nic_selection=Selection_NICTypes[module.params['nic_selection']])
    if module.params['failover_network'] is not None:
        idrac.config_mgr.configure_network_settings(failover_network=Failover_NICTypes[module.params['failover_network']])
    if module.params['auto_detect'] is not None:
        idrac.config_mgr.configure_network_settings(auto_detect=AutoDetect_NICTypes[module.params['auto_detect']])
    if module.params['auto_negotiation'] is not None:
        idrac.config_mgr.configure_network_settings(auto_negotiation=Autoneg_NICTypes[module.params['auto_negotiation']])
    if module.params['network_speed'] is not None:
        idrac.config_mgr.configure_network_settings(network_speed=Speed_NICTypes[module.params['network_speed']])
    if module.params['duplex_mode'] is not None:
        idrac.config_mgr.configure_network_settings(duplex_mode=Duplex_NICTypes[module.params['duplex_mode']])
    if module.params['nic_mtu'] is not None:
        idrac.config_mgr.configure_network_settings(nic_mtu=module.params['nic_mtu'])
    if module.params['enable_dhcp'] is not None:
        idrac.config_mgr.configure_ipv4(enable_dhcp=DHCPEnable_IPv4Types[module.params['enable_dhcp']])
    if module.params['ip_address'] is not None:
        idrac.config_mgr.configure_ipv4(ip_address=module.params['ip_address'])
    if module.params['enable_ipv4'] is not None:
        idrac.config_mgr.configure_ipv4(enable_ipv4=Enable_IPv4Types[module.params['enable_ipv4']])
    if module.params['dns_from_dhcp'] is not None:
        idrac.config_mgr.configure_static_ipv4(dns_from_dhcp=DNSFromDHCP_IPv4StaticTypes[module.params['dns_from_dhcp']])
    if module.params['static_dns_1'] is not None:
        idrac.config_mgr.configure_static_ipv4(dns_1=module.params['static_dns_1'])
    if module.params['static_dns_2'] is not None:
        idrac.config_mgr.configure_static_ipv4(dns_2=module.params['static_dns_2'])
    if module.params['static_gateway'] is not None:
        idrac.config_mgr.configure_static_ipv4(gateway=module.params['static_gateway'])
    if module.params['static_net_mask'] is not None:
        idrac.config_mgr.configure_static_ipv4(net_mask=module.params['static_net_mask'])
    if module.check_mode:
        msg = idrac.config_mgr.is_change_applicable()
    else:
        msg = idrac.config_mgr.apply_changes(reboot=False)
    return msg