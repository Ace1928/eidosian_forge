from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def list_by_automaiton_account(self):
    result = []
    try:
        resp = self.automation_client.runbook.list_by_automation_account(self.resource_group, self.automation_account_name)
        while True:
            result.append(resp.next())
    except StopIteration:
        pass
    except Exception as exc:
        pass
    return result