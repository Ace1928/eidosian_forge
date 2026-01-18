from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
def ensure_private_endpoint(self):
    try:
        self.network_client.private_endpoints.get(resource_group_name=self.resource_group, private_endpoint_name=self.private_endpoint)
    except ResourceNotFoundError:
        self.fail('Could not load the private endpoint {0}.'.format(self.private_endpoint))