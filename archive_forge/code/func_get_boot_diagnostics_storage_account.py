from __future__ import absolute_import, division, print_function
import base64
import random
import re
import time
from ansible.module_utils.basic import to_native, to_bytes
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import (AzureRMModuleBase,
def get_boot_diagnostics_storage_account(self, limited=False, vm_dict=None):
    """
        Get the boot diagnostics storage account.

        Arguments:
          - limited - if true, limit the logic to the boot_diagnostics storage account
                      this is used if initial creation of the VM has a stanza with
                      boot_diagnostics disabled, so we only create a storage account
                      if the user specifies a storage account name inside the boot_diagnostics
                      schema
          - vm_dict - if invoked on an update, this is the current state of the vm including
                      tags, like the default storage group tag '_own_sa_'.

        Normal behavior:
          - try the self.boot_diagnostics.storage_account field
          - if not there, try the self.storage_account_name field
          - if not there, use the default storage account

        If limited is True:
          - try the self.boot_diagnostics.storage_account field
          - if not there, None
        """
    bsa = None
    if self.boot_diagnostics is not None and self.boot_diagnostics.get('storage_account') is not None:
        if self.boot_diagnostics.get('resource_group') is not None:
            bsa = self.get_storage_account(self.boot_diagnostics['resource_group'], self.boot_diagnostics['storage_account'])
        else:
            bsa = self.get_storage_account(self.resource_group, self.boot_diagnostics['storage_account'])
    elif limited:
        return None
    elif self.storage_account_name:
        bsa = self.get_storage_account(self.resource_group, self.storage_account_name)
    else:
        bsa = self.create_default_storage_account(vm_dict=vm_dict)
    self.log('boot diagnostics storage account:')
    self.log(self.serialize_obj(bsa, 'StorageAccount'), pretty_print=True)
    return bsa