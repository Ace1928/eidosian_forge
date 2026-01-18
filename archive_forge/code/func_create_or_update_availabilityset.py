from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_or_update_availabilityset(self):
    """
        Method calling the Azure SDK to create or update the AS.
        :return: void
        """
    self.log('Creating availabilityset {0}'.format(self.name))
    try:
        params_sku = self.compute_models.Sku(name=self.sku)
        params = self.compute_models.AvailabilitySet(location=self.location, tags=self.tags, platform_update_domain_count=self.platform_update_domain_count, platform_fault_domain_count=self.platform_fault_domain_count, proximity_placement_group=self.proximity_placement_group_resource, sku=params_sku)
        response = self.compute_client.availability_sets.create_or_update(self.resource_group, self.name, params)
    except Exception as e:
        self.log('Error attempting to create the availability set.')
        self.fail('Error creating the availability set: {0}'.format(str(e)))
    return availability_set_to_dict(response)