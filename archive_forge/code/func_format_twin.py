from __future__ import absolute_import, division, print_function
import re
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def format_twin(self, item):
    if not item:
        return None
    format_twin = dict(device_id=item.device_id, module_id=item.module_id, tags=item.tags, properties=dict(), etag=item.etag, version=item.version, device_etag=item.device_etag, status=item.status, cloud_to_device_message_count=item.cloud_to_device_message_count, authentication_type=item.authentication_type)
    if item.properties is not None:
        format_twin['properties']['desired'] = item.properties.desired
        format_twin['properties']['reported'] = item.properties.reported
    return format_twin