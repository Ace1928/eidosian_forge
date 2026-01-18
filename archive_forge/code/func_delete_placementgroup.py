from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, normalize_location_name
def delete_placementgroup(self):
    try:
        response = self.compute_client.proximity_placement_groups.delete(resource_group_name=self.resource_group, proximity_placement_group_name=self.name)
    except Exception as exc:
        self.fail('Error deleting proximity placement group {0} - {1}'.format(self.name, str(exc)))
    return response