from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils.connection import ConnectionError
def get_interfaces_acl_bind_requests(self, commands):
    """Get requests to bind specified ACLs for all interfaces
        specified the commands
        """
    requests = []
    for command in commands:
        intf_name = command['name']
        url = self.acl_interfaces_path.format(intf_name=intf_name)
        for access_group in command['access_groups']:
            for acl in access_group['acls']:
                if acl['direction'] == 'in':
                    payload = {'openconfig-acl:config': {'id': intf_name}, 'openconfig-acl:interface-ref': {'config': {'interface': intf_name.split('.')[0]}}, 'openconfig-acl:ingress-acl-sets': {'ingress-acl-set': [{'set-name': acl['name'], 'type': acl_type_to_payload_map[access_group['type']], 'config': {'set-name': acl['name'], 'type': acl_type_to_payload_map[access_group['type']]}}]}}
                else:
                    payload = {'openconfig-acl:config': {'id': intf_name}, 'openconfig-acl:interface-ref': {'config': {'interface': intf_name.split('.')[0]}}, 'openconfig-acl:egress-acl-sets': {'egress-acl-set': [{'set-name': acl['name'], 'type': acl_type_to_payload_map[access_group['type']], 'config': {'set-name': acl['name'], 'type': acl_type_to_payload_map[access_group['type']]}}]}}
                if '.' in intf_name:
                    payload['openconfig-acl:interface-ref']['config']['subinterface'] = int(intf_name.split('.')[1])
                requests.append({'path': url, 'method': POST, 'data': payload})
    return requests