from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_update_devtestlabartifactssource(self):
    """
        Creates or updates DevTest Labs Artifacts Source with the specified configuration.

        :return: deserialized DevTest Labs Artifacts Source instance state dictionary
        """
    self.log('Creating / Updating the DevTest Labs Artifacts Source instance {0}'.format(self.name))
    try:
        response = self.mgmt_client.artifact_sources.create_or_update(resource_group_name=self.resource_group, lab_name=self.lab_name, name=self.name, artifact_source=self.artifact_source)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.log('Error attempting to create the DevTest Labs Artifacts Source instance.')
        self.fail('Error creating the DevTest Labs Artifacts Source instance: {0}'.format(str(exc)))
    return response.as_dict()