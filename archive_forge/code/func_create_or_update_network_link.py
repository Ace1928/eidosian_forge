from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, \
def create_or_update_network_link(self, virtual_network_link):
    try:
        response = self.private_dns_client.virtual_network_links.begin_create_or_update(resource_group_name=self.resource_group, private_zone_name=self.zone_name, virtual_network_link_name=self.name, parameters=virtual_network_link)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.fail('Error creating or updating virtual network link {0} - {1}'.format(self.name, str(exc)))
    return self.vnetlink_to_dict(response)