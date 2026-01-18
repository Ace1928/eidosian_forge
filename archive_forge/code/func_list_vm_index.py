from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, azure_id_to_dict
def list_vm_index(self):
    try:
        res = self.network_client.network_interfaces.list_virtual_machine_scale_set_vm_network_interfaces(resource_group_name=self.resource_group, virtual_machine_scale_set_name=self.vmss_name, virtualmachine_index=self.vm_index)
        return list(res)
    except Exception as exc:
        self.fail('Error listing by resource group {0} - {1}'.format(self.resource_group, str(exc)))