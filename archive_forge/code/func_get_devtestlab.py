from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
def get_devtestlab(self):
    """
        Gets the properties of the specified DevTest Lab.

        :return: deserialized DevTest Lab instance state dictionary
        """
    self.log('Checking if the DevTest Lab instance {0} is present'.format(self.lab_name))
    try:
        response = self.mgmt_client.labs.get(resource_group_name=self.resource_group, name=self.lab_name)
        self.log('Response : {0}'.format(response))
        self.log('DevTest Lab instance : {0} found'.format(response.name))
        return response.as_dict()
    except ResourceNotFoundError as e:
        self.fail('Did not find the DevTest Lab instance.')
        return False