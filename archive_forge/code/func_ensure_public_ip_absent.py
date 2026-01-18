from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def ensure_public_ip_absent(self, server_ids):
    """
        Ensures the given server ids having the public ip removed if there is any
        :param server_ids: the list of server ids
        :return: (changed, changed_server_ids, results)
                  changed: A flag indicating if there is any change
                  changed_server_ids : the list of server ids that are changed
                  results: The result list from clc public ip call
        """
    changed = False
    results = []
    changed_server_ids = []
    servers = self._get_servers_from_clc(server_ids, 'Failed to obtain server list from the CLC API')
    servers_to_change = [server for server in servers if len(server.PublicIPs().public_ips) > 0]
    for server in servers_to_change:
        if not self.module.check_mode:
            result = self._remove_publicip_from_server(server)
            results.append(result)
        changed_server_ids.append(server.id)
        changed = True
    return (changed, changed_server_ids, results)