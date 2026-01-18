from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel, _camel_to_snake
import re
def delete_hub(self):
    try:
        self.IoThub_client.iot_hub_resource.begin_delete(self.resource_group, self.name)
        return True
    except Exception as exc:
        self.fail('Error deleting IoT Hub {0}: {1}'.format(self.name, exc.message or str(exc)))
        return False