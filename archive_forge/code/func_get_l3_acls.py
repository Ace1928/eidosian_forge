from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.connection import ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.l3_acls.l3_acls import L3_aclsArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def get_l3_acls(self):
    """Get all l3 acl configurations available in chassis"""
    acls_path = 'data/openconfig-acl:acl/acl-sets'
    method = 'GET'
    request = [{'path': acls_path, 'method': method}]
    try:
        response = edit_config(self._module, to_request(self._module, request))
    except ConnectionError as exc:
        self._module.fail_json(msg=str(exc), code=exc.code)
    acls = []
    if response[0][1].get('openconfig-acl:acl-sets'):
        acls = response[0][1]['openconfig-acl:acl-sets'].get('acl-set', [])
    ipv4_acls_configs = []
    ipv6_acls_configs = []
    for acl in acls:
        is_ipv4 = False
        acl_config = {}
        acl_rules = []
        config = acl['config']
        if config.get('type') in ('ACL_IPV4', 'openconfig-acl:ACL_IPV4'):
            is_ipv4 = True
        elif config.get('type') in ('ACL_IPV6', 'openconfig-acl:ACL_IPV6'):
            is_ipv4 = False
        else:
            continue
        acl_config['name'] = config['name']
        acl_config['remark'] = config.get('description')
        acl_config['rules'] = acl_rules
        acl_entries = acl.get('acl-entries', {}).get('acl-entry', [])
        for acl_entry in acl_entries:
            acl_rule = {}
            acl_entry_config = acl_entry['config']
            acl_rule['sequence_num'] = acl_entry_config['sequence-id']
            acl_rule['remark'] = acl_entry_config.get('description')
            acl_rule['action'] = acl_entry['actions']['config']['forwarding-action']
            acl_rule['l2'] = acl_entry.get('l2', {}).get('config', {})
            if is_ipv4:
                acl_rule['l3'] = acl_entry.get('ipv4', {}).get('config', {})
            else:
                acl_rule['l3'] = acl_entry.get('ipv6', {}).get('config', {})
            acl_rule['l4'] = acl_entry.get('transport', {}).get('config', {})
            acl_rules.append(acl_rule)
        if is_ipv4:
            ipv4_acls_configs.append(acl_config)
        else:
            ipv6_acls_configs.append(acl_config)
    l3_acls_configs = []
    if ipv4_acls_configs:
        l3_acls_configs.append({'address_family': 'ipv4', 'acls': ipv4_acls_configs})
    if ipv6_acls_configs:
        l3_acls_configs.append({'address_family': 'ipv6', 'acls': ipv6_acls_configs})
    return l3_acls_configs