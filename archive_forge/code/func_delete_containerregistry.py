from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def delete_containerregistry(self):
    """
        Deletes the specified container registry in the specified subscription and resource group.

        :return: True
        """
    self.log('Deleting the container registry instance {0}'.format(self.name))
    try:
        poller = self.containerregistry_client.registries.begin_delete(resource_group_name=self.resource_group, registry_name=self.name)
        response = self.get_poller_result(poller)
        self.log('Delete container registry response: {0}'.format(response))
    except Exception as e:
        self.log('Error attempting to delete the container registry instance.')
        self.fail('Error deleting the container registry instance: {0}'.format(str(e)))
    return True