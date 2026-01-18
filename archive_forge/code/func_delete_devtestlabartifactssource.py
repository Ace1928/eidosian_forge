from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def delete_devtestlabartifactssource(self):
    """
        Deletes specified DevTest Labs Artifacts Source instance in the specified subscription and resource group.

        :return: True
        """
    self.log('Deleting the DevTest Labs Artifacts Source instance {0}'.format(self.name))
    try:
        response = self.mgmt_client.artifact_sources.delete(resource_group_name=self.resource_group, lab_name=self.lab_name, name=self.name)
    except Exception as e:
        self.log('Error attempting to delete the DevTest Labs Artifacts Source instance.')
        self.fail('Error deleting the DevTest Labs Artifacts Source instance: {0}'.format(str(e)))
    return True