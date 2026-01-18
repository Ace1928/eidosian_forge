from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
def create_update_setting(self):
    try:
        response = self.monitor_diagnostic_settings_client.diagnostic_settings.create_or_update(resource_uri=self.resource, name=self.name, parameters=self.parameters)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
        return self.diagnostic_setting_to_dict(response)
    except Exception as exc:
        self.fail('Error creating or updating diagnostic setting {0} for resource {1}: {2}'.format(self.name, self.resource, str(exc)))