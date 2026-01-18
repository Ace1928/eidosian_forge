from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_device_module(self):
    try:
        response = self.mgmt_client.get_module(self.name, self.module_id)
        return self.format_module(response)
    except Exception as exc:
        self.fail('Error when getting IoT Hub device {0}: {1}'.format(self.name, exc))