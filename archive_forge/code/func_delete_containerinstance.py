from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
def delete_containerinstance(self):
    """
        Deletes the specified container group instance in the specified subscription and resource group.

        :return: True
        """
    self.log('Deleting the container instance {0}'.format(self.name))
    try:
        response = self.containerinstance_client.container_groups.begin_delete(resource_group_name=self.resource_group, container_group_name=self.name)
        return True
    except Exception as exc:
        self.fail('Error when deleting ACI {0}: {1}'.format(self.name, exc.message or str(exc)))
        return False