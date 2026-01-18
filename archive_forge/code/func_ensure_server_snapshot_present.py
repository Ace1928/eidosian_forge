from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def ensure_server_snapshot_present(self, server_ids, expiration_days):
    """
        Ensures the given set of server_ids have the snapshots created
        :param server_ids: The list of server_ids to create the snapshot
        :param expiration_days: The number of days to keep the snapshot
        :return: (changed, request_list, changed_servers)
                 changed: A flag indicating whether any change was made
                 request_list: the list of clc request objects from CLC API call
                 changed_servers: The list of servers ids that are modified
        """
    request_list = []
    changed = False
    servers = self._get_servers_from_clc(server_ids, 'Failed to obtain server list from the CLC API')
    servers_to_change = [server for server in servers if len(server.GetSnapshots()) == 0]
    for server in servers_to_change:
        changed = True
        if not self.module.check_mode:
            request = self._create_server_snapshot(server, expiration_days)
            request_list.append(request)
    changed_servers = [server.id for server in servers_to_change if server.id]
    return (changed, request_list, changed_servers)