from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def aggregated_app_settings(self):
    """Combine both system and user app settings"""
    function_app_settings = self.necessary_functionapp_settings()
    for app_setting_key in self.app_settings:
        found_setting = None
        for s in function_app_settings:
            if s.name == app_setting_key:
                found_setting = s
                break
        if found_setting:
            found_setting.value = self.app_settings[app_setting_key]
        else:
            function_app_settings.append(NameValuePair(name=app_setting_key, value=self.app_settings[app_setting_key]))
    return function_app_settings