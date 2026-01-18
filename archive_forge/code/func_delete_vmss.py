from __future__ import absolute_import, division, print_function
import base64
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, azure_id_to_dict, format_resource_id
from ansible.module_utils.basic import to_native, to_bytes
def delete_vmss(self, vmss):
    self.log('Deleting virtual machine scale set {0}'.format(self.name))
    self.results['actions'].append('Deleted virtual machine scale set {0}'.format(self.name))
    try:
        poller = self.compute_client.virtual_machine_scale_sets.begin_delete(self.resource_group, self.name)
        self.get_poller_result(poller)
    except Exception as exc:
        self.fail('Error deleting virtual machine scale set {0} - {1}'.format(self.name, str(exc)))
    return True