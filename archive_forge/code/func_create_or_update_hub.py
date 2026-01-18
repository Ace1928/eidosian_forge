from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel, _camel_to_snake
import re
def create_or_update_hub(self, hub):
    try:
        poller = self.IoThub_client.iot_hub_resource.begin_create_or_update(self.resource_group, self.name, hub, if_match=hub.etag)
        return self.get_poller_result(poller)
    except Exception as exc:
        self.fail('Error creating or updating IoT Hub {0}: {1}'.format(self.name, exc.message or str(exc)))