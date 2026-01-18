from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def deallocate(self, instance_id):
    try:
        self.mgmt_client.virtual_machine_scale_set_vms.begin_deallocate(resource_group_name=self.resource_group, vm_scale_set_name=self.vmss_name, instance_id=instance_id)
    except Exception as e:
        self.log('Could not deallocate instance of Virtual Machine Scale Set VM.')
        self.fail('Could not deallocate instance of Virtual Machine Scale Set VM.')