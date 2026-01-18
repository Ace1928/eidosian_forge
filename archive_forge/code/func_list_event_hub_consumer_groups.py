from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _camel_to_snake
def list_event_hub_consumer_groups(self, resource_group, name, event_hub_endpoint='events'):
    result = []
    try:
        resp = self.IoThub_client.iot_hub_resource.list_event_hub_consumer_groups(resource_group, name, event_hub_endpoint)
        while True:
            cg = resp.next()
            result.append(dict(id=cg.id, name=cg.name))
    except StopIteration:
        pass
    except Exception as exc:
        self.fail('Failed to listing consumer group for IoT Hub {0}/{1} routing endpoint: {2}'.format(resource_group, name, str(exc)))
    return result