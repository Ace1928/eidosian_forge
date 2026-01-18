from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_addon_dict(addon):
    result = dict()
    addon = addon or dict()
    for key in addon.keys():
        result[key] = addon[key].config
        if result[key] is None:
            result[key] = {}
        result[key]['enabled'] = addon[key].enabled
    return result