from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def publish_runbook(self):
    response = None
    try:
        response = self.automation_client.runbook.begin_publish(self.resource_group, self.automation_account_name, self.name)
    except Exception as exc:
        self.fail('Error when updating automation account {0}: {1}'.format(self.name, exc.message))