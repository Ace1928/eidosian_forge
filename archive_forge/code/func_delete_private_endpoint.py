from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
def delete_private_endpoint(self):
    try:
        poller = self.network_client.private_endpoints.begin_delete(self.resource_group, self.name)
        result = self.get_poller_result(poller)
    except Exception as exc:
        self.fail('Error deleting private endpoint {0} - {1}'.format(self.name, str(exc)))
    return result