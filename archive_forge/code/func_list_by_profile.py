from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
import re
def list_by_profile(self):
    """Get all Azure Azure CDN endpoints within an Azure CDN profile"""
    self.log('List all Azure CDN endpoints within an Azure CDN profile')
    try:
        response = self.cdn_client.endpoints.list_by_profile(self.resource_group, self.profile_name)
    except Exception as exc:
        self.fail('Failed to list all items - {0}'.format(str(exc)))
    results = []
    for item in response:
        if self.has_tags(item.tags, self.tags):
            results.append(self.serialize_cdnendpoint(item))
    return results