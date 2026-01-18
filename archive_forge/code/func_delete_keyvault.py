from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def delete_keyvault(self):
    """
        Deletes specified Key Vault instance in the specified subscription and resource group.

        :return: True
        """
    self.log('Deleting the Key Vault instance {0}'.format(self.vault_name))
    try:
        response = self.mgmt_client.vaults.delete(resource_group_name=self.resource_group, vault_name=self.vault_name)
    except Exception as e:
        self.log('Error attempting to delete the Key Vault instance.')
        self.fail('Error deleting the Key Vault instance: {0}'.format(str(e)))
    return True