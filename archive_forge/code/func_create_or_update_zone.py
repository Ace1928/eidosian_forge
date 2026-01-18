from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
from ansible.module_utils._text import to_native
def create_or_update_zone(self, zone):
    try:
        new_zone = self.dns_client.zones.create_or_update(self.resource_group, self.name, zone)
    except Exception as exc:
        self.fail('Error creating or updating zone {0} - {1}'.format(self.name, exc.message or str(exc)))
    return zone_to_dict(new_zone)