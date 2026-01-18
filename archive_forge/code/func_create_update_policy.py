from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
def create_update_policy(self):
    """
        Creates or updates Policy with the specified configuration.

        :return: deserialized Policy instance state dictionary
        """
    self.log('Creating / Updating the Policy instance {0}'.format(self.name))
    try:
        response = self.mgmt_client.policies.create_or_update(resource_group_name=self.resource_group, lab_name=self.lab_name, policy_set_name=self.policy_set_name, name=self.name, policy=self.policy)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.log('Error attempting to create the Policy instance.')
        self.fail('Error creating the Policy instance: {0}'.format(str(exc)))
    return response.as_dict()