from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
def create_update_devtestlab(self):
    """
        Creates or updates DevTest Lab with the specified configuration.

        :return: deserialized DevTest Lab instance state dictionary
        """
    self.log('Creating / Updating the DevTest Lab instance {0}'.format(self.name))
    try:
        response = self.mgmt_client.labs.begin_create_or_update(resource_group_name=self.resource_group, name=self.name, lab=self.lab)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.log('Error attempting to create the DevTest Lab instance.')
        self.fail('Error creating the DevTest Lab instance: {0}'.format(str(exc)))
    return response.as_dict()