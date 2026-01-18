from __future__ import absolute_import, division, print_function
import copy
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AZURE_SUCCESS_STATE, AzureRMModuleBase
from ansible.module_utils._text import to_native
def check_name_availability(self):
    self.log('Checking name availability for {0}'.format(self.name))
    try:
        account_name = self.storage_models.StorageAccountCheckNameAvailabilityParameters(name=self.name)
        self.storage_client.storage_accounts.check_name_availability(account_name)
    except Exception as e:
        self.log('Error attempting to validate name.')
        self.fail('Error checking name availability: {0}'.format(str(e)))