from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_update_environment(self):
    """
        Creates or updates Environment with the specified configuration.

        :return: deserialized Environment instance state dictionary
        """
    self.log('Creating / Updating the Environment instance {0}'.format(self.name))
    try:
        if self.to_do == Actions.Create:
            response = self.mgmt_client.environments.begin_create_or_update(resource_group_name=self.resource_group, lab_name=self.lab_name, user_name=self.user_name, name=self.name, dtl_environment=self.dtl_environment)
        else:
            response = self.mgmt_client.environments.update(resource_group_name=self.resource_group, lab_name=self.lab_name, user_name=self.user_name, name=self.name, dtl_environment=self.dtl_environment)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.log('Error attempting to create the Environment instance.')
        self.fail('Error creating the Environment instance: {0}'.format(str(exc)))
    return response.as_dict()