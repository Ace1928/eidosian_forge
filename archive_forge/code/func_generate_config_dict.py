from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
from datetime import datetime
def generate_config_dict(blade):
    config_info = {}
    config_info['dns'] = blade.dns.list_dns().items[0].to_dict()
    config_info['smtp'] = blade.smtp.list_smtp().items[0].to_dict()
    try:
        config_info['alert_watchers'] = blade.alert_watchers.list_alert_watchers().items[0].to_dict()
    except Exception:
        config_info['alert_watchers'] = ''
    api_version = blade.api_version.list_versions().versions
    if HARD_LIMIT_API_VERSION in api_version:
        config_info['array_management'] = blade.directory_services.list_directory_services(names=['management']).items[0].to_dict()
        config_info['directory_service_roles'] = {}
        roles = blade.directory_services.list_directory_services_roles()
        for role in range(0, len(roles.items)):
            role_name = roles.items[role].name
            config_info['directory_service_roles'][role_name] = {'group': roles.items[role].group, 'group_base': roles.items[role].group_base}
    config_info['nfs_directory_service'] = blade.directory_services.list_directory_services(names=['nfs']).items[0].to_dict()
    config_info['smb_directory_service'] = blade.directory_services.list_directory_services(names=['smb']).items[0].to_dict()
    config_info['ntp'] = blade.arrays.list_arrays().items[0].ntp_servers
    config_info['ssl_certs'] = blade.certificates.list_certificates().items[0].to_dict()
    api_version = blade.api_version.list_versions().versions
    if CERT_GROUPS_API_VERSION in api_version:
        try:
            config_info['certificate_groups'] = blade.certificate_groups.list_certificate_groups().items[0].to_dict()
        except Exception:
            config_info['certificate_groups'] = ''
    if REPLICATION_API_VERSION in api_version:
        config_info['snmp_agents'] = {}
        snmp_agents = blade.snmp_agents.list_snmp_agents()
        for agent in range(0, len(snmp_agents.items)):
            agent_name = snmp_agents.items[agent].name
            config_info['snmp_agents'][agent_name] = {'version': snmp_agents.items[agent].version, 'engine_id': snmp_agents.items[agent].engine_id}
            if config_info['snmp_agents'][agent_name]['version'] == 'v3':
                config_info['snmp_agents'][agent_name]['auth_protocol'] = snmp_agents.items[agent].v3.auth_protocol
                config_info['snmp_agents'][agent_name]['privacy_protocol'] = snmp_agents.items[agent].v3.privacy_protocol
                config_info['snmp_agents'][agent_name]['user'] = snmp_agents.items[agent].v3.user
        config_info['snmp_managers'] = {}
        snmp_managers = blade.snmp_managers.list_snmp_managers()
        for manager in range(0, len(snmp_managers.items)):
            mgr_name = snmp_managers.items[manager].name
            config_info['snmp_managers'][mgr_name] = {'version': snmp_managers.items[manager].version, 'host': snmp_managers.items[manager].host, 'notification': snmp_managers.items[manager].notification}
            if config_info['snmp_managers'][mgr_name]['version'] == 'v3':
                config_info['snmp_managers'][mgr_name]['auth_protocol'] = snmp_managers.items[manager].v3.auth_protocol
                config_info['snmp_managers'][mgr_name]['privacy_protocol'] = snmp_managers.items[manager].v3.privacy_protocol
                config_info['snmp_managers'][mgr_name]['user'] = snmp_managers.items[manager].v3.user
    return config_info