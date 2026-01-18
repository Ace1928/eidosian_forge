from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import _load_params
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, HAS_AZURE
def delete_record_set(self):
    try:
        self.dns_client.record_sets.delete(resource_group_name=self.resource_group, zone_name=self.zone_name, relative_record_set_name=self.relative_name, record_type=self.record_type)
    except Exception as exc:
        self.fail('Error deleting record set {0} - {1}'.format(self.relative_name, exc.message or str(exc)))
    return None