from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_keyvault_client(self):
    return SecretClient(vault_url=self.keyvault_uri, credential=self.azure_auth.azure_credential_track2)