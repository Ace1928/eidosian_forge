from __future__ import absolute_import, division, print_function
import re
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def list_device_modules(self):
    try:
        response = None
        response = self.mgmt_client.get_modules(self.name)
        response = [self.format_item(item) for item in response]
        return response
    except Exception as exc:
        if hasattr(exc, 'message'):
            pass
        else:
            self.fail('Error when listing IoT Hub devices in {0}: {1}'.format(self.hub, exc))