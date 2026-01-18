from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.six.moves.urllib.parse import urlparse
import re
def list_items_by_resourcegroup(self):
    self.log('List all items')
    try:
        items = self.compute_client.virtual_machines.list(self.resource_group)
    except ResourceNotFoundError as exc:
        self.fail('Failed to list all items - {0}'.format(str(exc)))
    results = []
    for item in items:
        if self.has_tags(item.tags, self.tags):
            results.append(self.get_vm(self.resource_group, item.name))
    return results