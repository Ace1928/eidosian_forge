from __future__ import absolute_import, division, print_function
import base64
import random
import re
import time
from ansible.module_utils.basic import to_native, to_bytes
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import (AzureRMModuleBase,
def check_storage_account_name(self, name):
    self.log('Checking storage account name availability for {0}'.format(name))
    try:
        account_name = self.storage_models.StorageAccountCheckNameAvailabilityParameters(name=name)
        response = self.storage_client.storage_accounts.check_name_availability(account_name)
        if response.reason == 'AccountNameInvalid':
            raise Exception('Invalid default storage account name: {0}'.format(name))
    except Exception as exc:
        self.fail('Error checking storage account name availability for {0} - {1}'.format(name, str(exc)))
    return response.name_available