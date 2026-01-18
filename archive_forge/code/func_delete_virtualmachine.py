from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
def delete_virtualmachine(self):
    """
        Deletes specified Virtual Machine instance in the specified subscription and resource group.

        :return: True
        """
    self.log('Deleting the Virtual Machine instance {0}'.format(self.name))
    try:
        response = self.mgmt_client.virtual_machines.begin_delete(resource_group_name=self.resource_group, lab_name=self.lab_name, name=self.name)
    except Exception as e:
        self.log('Error attempting to delete the Virtual Machine instance.')
        self.fail('Error deleting the Virtual Machine instance: {0}'.format(str(e)))
    if isinstance(response, LROPoller):
        response = self.get_poller_result(response)
    return True