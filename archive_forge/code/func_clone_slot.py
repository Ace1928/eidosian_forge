from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def clone_slot(self):
    if self.configuration_source:
        src_slot = None if self.configuration_source.lower() == self.webapp_name.lower() else self.configuration_source
        if src_slot is None:
            site_config_clone_from = self.get_configuration()
        else:
            site_config_clone_from = self.get_configuration_slot(slot_name=src_slot)
        self.update_configuration_slot(site_config=site_config_clone_from)
        if src_slot is None:
            app_setting_clone_from = self.list_app_settings()
        else:
            app_setting_clone_from = self.list_app_settings_slot(src_slot)
        if self.app_settings:
            app_setting_clone_from.update(self.app_settings)
        self.update_app_settings_slot(app_settings=app_setting_clone_from)