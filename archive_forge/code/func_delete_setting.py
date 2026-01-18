from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
def delete_setting(self):
    try:
        response = self.monitor_diagnostic_settings_client.diagnostic_settings.delete(resource_uri=self.resource, name=self.name)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
        return response
    except Exception as exc:
        self.fail('Error deleting diagnostic setting {0} for resource {1}: {2}'.format(self.name, self.resource, str(exc)))