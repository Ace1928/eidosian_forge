from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel, _camel_to_snake
import re
def lookup_ip_filter(self, target, ip_filters):
    if not ip_filters or len(ip_filters) == 0:
        return False
    for item in ip_filters:
        if item.filter_name == target['name']:
            if item.ip_mask != target['ip_mask']:
                return False
            if item.action.lower() != target['action']:
                return False
            return True
    return False