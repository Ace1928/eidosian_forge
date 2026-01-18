from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.online import (
def all_detailed_servers(self):
    servers_api_path = self.get_resources()
    server_data = (self._get_server_detail(server_api_path) for server_api_path in servers_api_path)
    return [s for s in server_data if s is not None]