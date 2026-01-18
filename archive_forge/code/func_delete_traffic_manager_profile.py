from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def delete_traffic_manager_profile(self):
    """
        Deletes the specified Traffic Manager profile in the specified subscription and resource group.
        :return: True
        """
    self.log('Deleting the Traffic Manager profile {0}'.format(self.name))
    try:
        operation_result = self.traffic_manager_management_client.profiles.delete(self.resource_group, self.name)
        return True
    except Exception as e:
        self.log('Error attempting to delete the Traffic Manager profile.')
        self.fail('Error deleting the Traffic Manager profile: {0}'.format(e.message))
        return False