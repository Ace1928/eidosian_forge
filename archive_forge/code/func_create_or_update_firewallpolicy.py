from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, \
import copy
def create_or_update_firewallpolicy(self, firewall_policy):
    try:
        response = self.network_client.firewall_policies.begin_create_or_update(resource_group_name=self.resource_group, firewall_policy_name=self.name, parameters=firewall_policy)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.fail('Error creating or updating Firewall policy {0} - {1}'.format(self.name, str(exc)))
    return self.firewallpolicy_to_dict(response)