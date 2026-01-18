from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
import uuid
import datetime
def key_exists(self, old_passwords):
    for pd in old_passwords:
        if pd.key_id == self.key_id:
            return True
    return False