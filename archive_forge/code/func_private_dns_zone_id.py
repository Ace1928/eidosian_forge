from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
def private_dns_zone_id(self, name):
    return resource_id(subscription=self.subscription_id, resource_group=self.resource_group, namespace='Microsoft.Network', type='privateDnsZones', name=name)