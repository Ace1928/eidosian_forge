from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def list_items_by_location(self):
    self.log('List items by location')
    try:
        items = self.compute_client.virtual_machine_sizes.list(location=self.location)
    except ResourceNotFoundError as exc:
        self.fail('Failed to list items - {0}'.format(str(exc)))
    return [self.serialize_size(item) for item in items if self.name is None or self.name == item.name]