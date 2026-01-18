from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
from ansible.module_utils._text import to_native
def create_or_update_load_balancer(self, param):
    try:
        poller = self.network_client.load_balancers.begin_create_or_update(self.resource_group, self.name, param)
        new_lb = self.get_poller_result(poller)
        return new_lb
    except Exception as exc:
        self.fail('Error creating or updating load balancer {0} - {1}'.format(self.name, str(exc)))