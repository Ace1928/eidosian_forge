from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
import time
def event_hub_to_dict(item):
    event_hub = item.as_dict()
    result = dict()
    if item.additional_properties:
        result['additional_properties'] = item.additional_properties
    result['name'] = event_hub.get('name', None)
    result['partition_ids'] = event_hub.get('partition_ids', None)
    result['created_at'] = event_hub.get('created_at', None)
    result['updated_at'] = event_hub.get('updated_at', None)
    result['message_retention_in_days'] = event_hub.get('message_retention_in_days', None)
    result['partition_count'] = event_hub.get('partition_count', None)
    result['status'] = event_hub.get('status', None)
    result['tags'] = event_hub.get('tags', None)
    return result