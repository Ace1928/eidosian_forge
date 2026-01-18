from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_addon_profile_instance(self, addon):
    result = dict()
    addon = addon or {}
    for key in addon.keys():
        if not ADDONS.get(key):
            self.fail('Unsupported addon {0}'.format(key))
        if addon.get(key):
            name = ADDONS[key]['name']
            config_spec = ADDONS[key].get('config') or dict()
            config = addon[key]
            for v in config_spec.keys():
                config[config_spec[v]] = config[v]
            result[name] = self.managedcluster_models.ManagedClusterAddonProfile(config=config, enabled=config['enabled'])
    return result