from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _camel_to_snake
def instance_dict_to_dict(self, instance_dict):
    result = dict()
    for key in instance_dict.keys():
        result[key] = instance_dict[key].as_dict()
    return result