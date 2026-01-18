from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def delete_cg(self):
    try:
        return self.IoThub_client.iot_hub_resource.delete_event_hub_consumer_group(self.resource_group, self.hub, self.event_hub, self.name)
    except Exception as exc:
        self.fail('Error when deleting the consumer group {0} for IoT Hub {1} event hub {2}: {3}'.format(self.name, self.hub, self.event_hub, str(exc)))