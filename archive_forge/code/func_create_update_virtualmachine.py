from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
def create_update_virtualmachine(self):
    """
        Creates or updates Virtual Machine with the specified configuration.

        :return: deserialized Virtual Machine instance state dictionary
        """
    self.log('Creating / Updating the Virtual Machine instance {0}'.format(self.name))
    try:
        response = self.mgmt_client.virtual_machines.begin_create_or_update(resource_group_name=self.resource_group, lab_name=self.lab_name, name=self.name, lab_virtual_machine=self.lab_virtual_machine)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.log('Error attempting to create the Virtual Machine instance.')
        self.fail('Error creating the Virtual Machine instance: {0}'.format(str(exc)))
    return response.as_dict()