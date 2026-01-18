from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_update_keyvault(self):
    """
        Creates or updates Key Vault with the specified configuration.

        :return: deserialized Key Vault instance state dictionary
        """
    self.log('Creating / Updating the Key Vault instance {0}'.format(self.vault_name))
    try:
        response = self.mgmt_client.vaults.begin_create_or_update(resource_group_name=self.resource_group, vault_name=self.vault_name, parameters=self.parameters)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.log('Error attempting to create the Key Vault instance.')
        self.fail('Error creating the Key Vault instance: {0}'.format(str(exc)))
    return response.as_dict()