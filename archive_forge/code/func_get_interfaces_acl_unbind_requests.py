from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils.connection import ConnectionError
def get_interfaces_acl_unbind_requests(self, commands):
    """Get requests to unbind specified ACLs for all interfaces
        specified in the commands
        """
    requests = []
    for command in commands:
        intf_name = command['name']
        if not command.get('access_groups'):
            url = self.acl_interfaces_path.format(intf_name=intf_name)
            requests.append({'path': url, 'method': DELETE})
        else:
            for access_group in command['access_groups']:
                for acl in access_group['acls']:
                    if acl['direction'] == 'in':
                        url = self.ingress_acl_set_path.format(intf_name=intf_name, acl_name=acl['name'], acl_type=acl_type_to_payload_map[access_group['type']])
                        requests.append({'path': url, 'method': DELETE})
                    else:
                        url = self.egress_acl_set_path.format(intf_name=intf_name, acl_name=acl['name'], acl_type=acl_type_to_payload_map[access_group['type']])
                        requests.append({'path': url, 'method': DELETE})
    return requests