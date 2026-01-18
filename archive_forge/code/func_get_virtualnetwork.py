from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_virtualnetwork(self):
    """
        Gets the properties of the specified Virtual Network.

        :return: deserialized Virtual Network instance state dictionary
        """
    self.log('Checking if the Virtual Network instance {0} is present'.format(self.name))
    found = False
    try:
        response = self.mgmt_client.virtual_networks.get(resource_group_name=self.resource_group, lab_name=self.lab_name, name=self.name)
        found = True
        self.log('Response : {0}'.format(response))
        self.log('Virtual Network instance : {0} found'.format(response.name))
    except ResourceNotFoundError as e:
        self.log('Did not find the Virtual Network instance.')
    if found is True:
        return response.as_dict()
    return False