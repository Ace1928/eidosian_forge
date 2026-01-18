from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel, _camel_to_snake
import re
def construct_ip_filters(self):
    return [self.IoThub_models.IpFilterRule(filter_name=x['name'], action=self.IoThub_models.IpFilterActionType[x['action']], ip_mask=x['ip_mask']) for x in self.ip_filters]