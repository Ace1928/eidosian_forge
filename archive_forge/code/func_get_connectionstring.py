from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils._text import to_native
def get_connectionstring(self, resource_group, name):
    keys = ['', '']
    if not self.show_connection_string:
        return keys
    try:
        cred = self.storage_client.storage_accounts.list_keys(resource_group, name)
        try:
            keys = [cred.keys[0].value, cred.keys[1].value]
        except AttributeError:
            keys = [cred.key1, cred.key2]
    except Exception:
        pass
    return keys