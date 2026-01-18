from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def list_subscription(self):
    self.log('List items for subscription')
    try:
        response = self.network_client.ddos_protection_plans.list()
    except ResourceNotFoundError as exc:
        self.fail('Failed to list DDoS protection plan in the subscription - {0}'.format(str(exc)))
    results = []
    for item in response:
        if self.has_tags(item.tags, self.tags):
            results.append(item)
    return results