from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_detachment_configuration(self, vm_instance, disks_names):
    vm_data = parse_resource_id(vm_instance.id)
    leftovers = [d for d in vm_instance.storage_profile.data_disks if d.name.lower() not in disks_names]
    if len(vm_instance.storage_profile.data_disks) == len(leftovers):
        self.fail("None of the following disks '{0}' are attached to the VM '{1}/{2}'.".format(disks_names, vm_data['resource_group'], vm_data['resource_name']))
    vm_instance.storage_profile.data_disks = leftovers
    return (vm_data['resource_group'], vm_data['resource_name'], vm_instance)