from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, \
import copy
def delete_firewallpolicy(self):
    try:
        response = self.network_client.firewall_policies.begin_delete(resource_group_name=self.resource_group, firewall_policy_name=self.name)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.fail('Error deleting Firewall policy {0} - {1}'.format(self.name, str(exc)))
    return response