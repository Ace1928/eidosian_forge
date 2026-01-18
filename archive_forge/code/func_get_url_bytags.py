from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_rest import GenericRestClient
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
import json
def get_url_bytags(self):
    return '/subscriptions' + '/' + self.subscription_id + '/resourceGroups' + '/' + self.resource_group + '/providers' + '/Microsoft.ApiManagement' + '/service' + '/' + self.service_name + '/apisByTags'