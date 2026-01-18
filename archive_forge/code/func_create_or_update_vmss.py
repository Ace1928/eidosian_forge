from __future__ import absolute_import, division, print_function
import base64
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, azure_id_to_dict, format_resource_id
from ansible.module_utils.basic import to_native, to_bytes
def create_or_update_vmss(self, params):
    try:
        poller = self.compute_client.virtual_machine_scale_sets.begin_create_or_update(self.resource_group, self.name, params)
        self.get_poller_result(poller)
    except Exception as exc:
        self.fail('Error creating or updating virtual machine {0} - {1}'.format(self.name, str(exc)))