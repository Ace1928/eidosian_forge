from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, CIDR_PATTERN, azure_id_to_dict, format_resource_id
def create_or_update_subnet(self, subnet):
    try:
        poller = self.network_client.subnets.begin_create_or_update(self.resource_group, self.virtual_network_name, self.name, subnet)
        new_subnet = self.get_poller_result(poller)
    except Exception as exc:
        self.fail('Error creating or updating subnet {0} - {1}'.format(self.name, str(exc)))
    self.check_provisioning_state(new_subnet)
    return subnet_to_dict(new_subnet)