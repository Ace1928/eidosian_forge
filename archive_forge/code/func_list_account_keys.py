from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def list_account_keys(self, resource_group, name):
    try:
        resp = self.automation_client.keys.list_by_automation_account(resource_group, name)
        return [x.as_dict() for x in resp.keys]
    except Exception as exc:
        self.fail('Error when listing keys for automation account {0}/{1}: {2}'.format(resource_group, name, exc.message))