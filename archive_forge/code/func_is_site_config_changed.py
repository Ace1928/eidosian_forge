from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def is_site_config_changed(self, existing_config):
    for updatable_property in self.site_config_updatable_properties:
        if self.site_config.get(updatable_property):
            if not getattr(existing_config, updatable_property) or str(getattr(existing_config, updatable_property)).upper() != str(self.site_config.get(updatable_property)).upper():
                return True
    return False