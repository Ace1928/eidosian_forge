from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def list_by_name(self):
    self.log('Get app service plan {0}'.format(self.name))
    item = None
    result = []
    try:
        item = self.web_client.app_service_plans.get(resource_group_name=self.resource_group, name=self.name)
    except ResourceNotFoundError:
        pass
    if item and self.has_tags(item.tags, self.tags):
        curated_result = self.construct_curated_plan(item)
        result = [curated_result]
    return result