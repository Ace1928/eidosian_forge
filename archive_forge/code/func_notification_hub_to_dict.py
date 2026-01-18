from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def notification_hub_to_dict(item):
    notification_hub = item.as_dict()
    result = dict(additional_properties=notification_hub.get('additional_properties', {}), id=notification_hub.get('id', None), name=notification_hub.get('name', None), type=notification_hub.get('type', None), location=notification_hub.get('location', '').replace(' ', '').lower(), tags=notification_hub.get('tags', None), provisioning_state=notification_hub.get('provisioning_state', None), region=notification_hub.get('region', None), metric_id=notification_hub.get('metric_id', None), service_bus_endpoint=notification_hub.get('service_bus_endpoint', None), scale_unit=notification_hub.get('scale_unit', None), enabled=notification_hub.get('enabled', None), critical=notification_hub.get('critical', None), data_center=notification_hub.get('data_center', None), namespace_type=notification_hub.get('namespace_type', None))
    return result