from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
def process_parameters(self):
    if isinstance(self.resource, dict):
        if '/' not in self.resource.get('type'):
            self.fail("resource type parameter must include namespace, such as 'Microsoft.Network/virtualNetworks'")
        self.resource = resource_id(subscription=self.resource.get('subscription_id', self.subscription_id), resource_group=self.resource.get('resource_group'), namespace=self.resource.get('type').split('/')[0], type=self.resource.get('type').split('/')[1], name=self.resource.get('name'))