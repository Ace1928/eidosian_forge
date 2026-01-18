from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def compare_addon(origin, patch, config):
    if not patch:
        return True
    if not origin:
        return False
    if origin['enabled'] != patch['enabled']:
        return False
    config = config or dict()
    for key in config.keys():
        if origin.get(config[key]) != patch.get(key):
            return False
    return True