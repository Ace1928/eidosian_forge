from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
def delete_customimage(self):
    """
        Deletes specified Custom Image instance in the specified subscription and resource group.

        :return: True
        """
    self.log('Deleting the Custom Image instance {0}'.format(self.name))
    try:
        response = self.mgmt_client.custom_images.begin_delete(resource_group_name=self.resource_group, lab_name=self.lab_name, name=self.name)
    except Exception as e:
        self.log('Error attempting to delete the Custom Image instance.')
        self.fail('Error deleting the Custom Image instance: {0}'.format(str(e)))
    return True