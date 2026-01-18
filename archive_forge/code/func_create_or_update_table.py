from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, normalize_location_name
def create_or_update_table(self, param):
    try:
        poller = self.network_client.route_tables.begin_create_or_update(self.resource_group, self.name, param)
        return self.get_poller_result(poller)
    except Exception as exc:
        self.fail('Error creating or updating route table {0} - {1}'.format(self.name, str(exc)))