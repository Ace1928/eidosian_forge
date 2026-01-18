from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def detach_disks_from_all_vms(self, disk_instances):
    changed = False
    unique_vm_id = []
    for param, disk_instance in disk_instances:
        managed_by_vm = disk_instance.get('managed_by')
        managed_by_extended_vms = disk_instance.get('managed_by_extended') or []
        if managed_by_vm is not None and managed_by_vm not in unique_vm_id:
            unique_vm_id.append(managed_by_vm)
        for vm_id in managed_by_extended_vms:
            if vm_id not in unique_vm_id:
                unique_vm_id.append(vm_id)
    if unique_vm_id:
        disks_names = [instance.get('name').lower() for d, instance in disk_instances]
        changed = True
        attach_config = []
        for vm_id in unique_vm_id:
            vm_name_id = parse_resource_id(vm_id)
            vm_instance = self._get_vm(vm_name_id['resource_group'], vm_name_id['resource_name'])
            attach_config.append(self.create_detachment_configuration(vm_instance, disks_names))
        if len(attach_config) > 0:
            changed = True
            self.update_virtual_machines(attach_config)
    return changed