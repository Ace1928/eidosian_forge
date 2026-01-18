from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, \
def delete_network_link(self):
    try:
        response = self.private_dns_client.virtual_network_links.begin_delete(resource_group_name=self.resource_group, private_zone_name=self.zone_name, virtual_network_link_name=self.name)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.fail('Error deleting virtual network link {0} - {1}'.format(self.name, str(exc)))
    return response