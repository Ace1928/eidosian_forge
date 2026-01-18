from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def list_by_elastic_pool(self):
    response = None
    results = []
    try:
        response = self.sql_client.databases.list_by_elastic_pool(resource_group_name=self.resource_group, server_name=self.server_name, elastic_pool_name=self.elastic_pool_name)
        self.log('Response : {0}'.format(response))
    except Exception:
        self.fail('Could not get facts for Databases.')
    if response is not None:
        for item in response:
            if self.has_tags(item.tags, self.tags):
                results.append(self.format_item(item))
    return results