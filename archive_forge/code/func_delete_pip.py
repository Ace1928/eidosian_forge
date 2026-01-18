from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils._text import to_native
def delete_pip(self):
    try:
        poller = self.network_client.public_ip_addresses.begin_delete(self.resource_group, self.name)
        self.get_poller_result(poller)
    except Exception as exc:
        self.fail('Error deleting {0} - {1}'.format(self.name, str(exc)))
    self.results['state']['status'] = 'Deleted'
    return True