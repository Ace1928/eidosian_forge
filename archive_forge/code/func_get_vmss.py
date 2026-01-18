from __future__ import absolute_import, division, print_function
import base64
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, azure_id_to_dict, format_resource_id
from ansible.module_utils.basic import to_native, to_bytes
def get_vmss(self):
    """
        Get the VMSS

        :return: VirtualMachineScaleSet object
        """
    try:
        vmss = self.compute_client.virtual_machine_scale_sets.get(self.resource_group, self.name)
        return vmss
    except ResourceNotFoundError as exc:
        self.fail('Error getting virtual machine scale set {0} - {1}'.format(self.name, str(exc)))