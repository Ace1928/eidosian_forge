from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def list_by_subscription(self):
    result = []
    self.log('List all in by subscription')
    try:
        response = self.network_client.private_link_services.list_by_subscription()
        while True:
            result.append(response.next())
    except StopIteration:
        pass
    except Exception:
        pass
    return [self.service_to_dict(item) for item in result]