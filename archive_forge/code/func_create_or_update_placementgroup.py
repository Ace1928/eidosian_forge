from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, normalize_location_name
def create_or_update_placementgroup(self, proximity_placement_group):
    try:
        response = self.compute_client.proximity_placement_groups.create_or_update(resource_group_name=self.resource_group, proximity_placement_group_name=self.name, parameters=proximity_placement_group)
    except Exception as exc:
        self.fail('Error creating or updating proximity placement group {0} - {1}'.format(self.name, str(exc)))
    return self.ppg_to_dict(response)