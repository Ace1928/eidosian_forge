from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
import re
def fetch_config_subset(info_subset):
    if info_subset is None:
        return ({}, True)
    toreturn = {'config': {}}
    result = {}
    temp_dict = {}
    grp_fields = '\n        smtp_server,\n        smtp_port,\n        smtp_auth_enabled,\n        smtp_auth_username,\n        autosupport_enabled,\n        send_alert_to_support,\n        isns_enabled,\n        snmp_trap_enabled,\n        snmp_trap_host,\n        snmp_trap_port,\n        snmp_community,\n        domain_name,\n        dns_servers,\n        ntp_server,\n        syslogd_enabled,\n        syslogd_server,\n        vvol_enabled,\n        alarms_enabled,\n        member_list,\n        encryption_config,\n        name,\n        fc_enabled,\n        iscsi_enabled\n\n    '
    try:
        for key, cl_obj in info_subset.items():
            if key == 'arrays':
                resp = cl_obj.list(detail=True, fields='extended_model,full_name,all_flash,serial,role')
            elif key == 'groups':
                resp = cl_obj.list(detail=True, fields=re.sub('\\s+', '', grp_fields))
            elif key == 'pools':
                resp = cl_obj.list(detail=True, fields='array_count,dedupe_all_volumes,dedupe_capable,is_default,name,vol_list')
            elif key == 'network_configs':
                resp = cl_obj.list(detail=True)
            else:
                continue
            temp_dict[key] = resp
        result['arrays'] = generate_dict('arrays', temp_dict['arrays'])['arrays']
        result['groups'] = generate_dict('groups', temp_dict['groups'])['groups']
        result['pools'] = generate_dict('pools', temp_dict['pools'])['pools']
        result['network_configs'] = generate_dict('network_configs', temp_dict['network_configs'])['network_configs']
        toreturn['config'] = result
        return (toreturn, True)
    except Exception:
        raise