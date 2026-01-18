from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_update_replication(self):
    """
        Creates or updates Replication with the specified configuration.

        :return: deserialized Replication instance state dictionary
        """
    self.log('Creating / Updating the Replication instance {0}'.format(self.replication_name))
    try:
        if self.to_do == Actions.Create:
            replication = Replication(location=self.location)
            response = self.containerregistry_client.replications.begin_create(resource_group_name=self.resource_group, registry_name=self.registry_name, replication_name=self.replication_name, replication=replication)
        else:
            update_params = ReplicationUpdateParameters()
            response = self.containerregistry_client.replications.begin_update(resource_group_name=self.resource_group, registry_name=self.registry_name, replication_name=self.replication_name, replication_update_parameters=update_params)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.log('Error attempting to create the Replication instance.')
        self.fail('Error creating the Replication instance: {0}'.format(str(exc)))
    return create_replication_dict(response)