from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.connection import ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.acl_interfaces.acl_interfaces import Acl_interfacesArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def get_acl_interfaces(self):
    """Get all interface access-group configurations available in chassis"""
    acl_interfaces_path = 'data/openconfig-acl:acl/interfaces'
    method = 'GET'
    request = [{'path': acl_interfaces_path, 'method': method}]
    try:
        response = edit_config(self._module, to_request(self._module, request))
    except ConnectionError as exc:
        self._module.fail_json(msg=str(exc), code=exc.code)
    acl_interfaces = []
    if response[0][1].get('openconfig-acl:interfaces'):
        acl_interfaces = response[0][1]['openconfig-acl:interfaces'].get('interface', [])
    acl_interfaces_configs = {}
    for interface in acl_interfaces:
        acls_list = []
        ingress_acls = interface.get('ingress-acl-sets', {}).get('ingress-acl-set', [])
        for acl in ingress_acls:
            if acl.get('config'):
                acls_list.append({'name': acl['config']['set-name'], 'type': acl['config']['type'], 'direction': 'in'})
        egress_acls = interface.get('egress-acl-sets', {}).get('egress-acl-set', [])
        for acl in egress_acls:
            if acl.get('config'):
                acls_list.append({'name': acl['config']['set-name'], 'type': acl['config']['type'], 'direction': 'out'})
        acl_interfaces_configs[interface['id']] = acls_list
    return acl_interfaces_configs