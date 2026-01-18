from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_cg(self):
    try:
        return self.IoThub_client.iot_hub_resource.get_event_hub_consumer_group(self.resource_group, self.hub, self.event_hub, self.name)
    except Exception:
        pass
        return None