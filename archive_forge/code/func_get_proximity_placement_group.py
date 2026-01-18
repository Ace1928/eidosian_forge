from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_proximity_placement_group(self, resource_group, name):
    try:
        return self.compute_client.proximity_placement_groups.get(resource_group, name)
    except ResourceNotFoundError as exc:
        self.fail('Error fetching proximity placement group {0} - {1}'.format(name, str(exc)))