from __future__ import absolute_import, division, print_function
import base64
import random
import re
import time
from ansible.module_utils.basic import to_native, to_bytes
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import (AzureRMModuleBase,
def create_default_storage_account(self, vm_dict=None):
    """
        Create (once) a default storage account <vm name>XXXX, where XXXX is a random number.
        NOTE: If <vm name>XXXX exists, use it instead of failing.  Highly unlikely.
        If this method is called multiple times across executions it will return the same
        storage account created with the random name which is stored in a tag on the VM.

        vm_dict is passed in during an update, so we can obtain the _own_sa_ tag and return
        the default storage account we created in a previous invocation

        :return: storage account object
        """
    account = None
    valid_name = False
    if self.tags is None:
        self.tags = {}
    if self.tags.get('_own_sa_', None):
        return self.get_storage_account(self.resource_group, self.tags['_own_sa_'])
    if vm_dict and vm_dict.get('tags', {}).get('_own_sa_', None):
        return self.get_storage_account(self.resource_group, vm_dict['tags']['_own_sa_'])
    storage_account_name_base = re.sub('[^a-zA-Z0-9]', '', self.name[:20].lower())
    for i in range(0, 5):
        rand = random.randrange(1000, 9999)
        storage_account_name = storage_account_name_base + str(rand)
        if self.check_storage_account_name(storage_account_name):
            valid_name = True
            break
    if not valid_name:
        self.fail('Failed to create a unique storage account name for {0}. Try using a different VM name.'.format(self.name))
    try:
        account = self.storage_client.storage_accounts.get_properties(self.resource_group, storage_account_name)
    except Exception:
        pass
    if account:
        self.log('Storage account {0} found.'.format(storage_account_name))
        self.check_provisioning_state(account)
        return account
    sku = self.storage_models.Sku(name=self.storage_models.SkuName.standard_lrs)
    sku.tier = self.storage_models.SkuTier.standard
    kind = self.storage_models.Kind.storage
    parameters = self.storage_models.StorageAccountCreateParameters(sku=sku, kind=kind, location=self.location)
    self.log('Creating storage account {0} in location {1}'.format(storage_account_name, self.location))
    self.results['actions'].append('Created storage account {0}'.format(storage_account_name))
    try:
        poller = self.storage_client.storage_accounts.begin_create(self.resource_group, storage_account_name, parameters)
        self.get_poller_result(poller)
    except Exception as exc:
        self.fail('Failed to create storage account: {0} - {1}'.format(storage_account_name, str(exc)))
    self.tags['_own_sa_'] = storage_account_name
    return self.get_storage_account(self.resource_group, storage_account_name)