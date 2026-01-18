from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def detach_or_delete_disks(self, managed_vm_id):
    changed, disk_instances = (False, [])
    for disk in self.managed_disks:
        params, disk_instance = self.get_disk_instance(disk)
        if disk_instance is not None:
            disk_instances.append((disk, disk_instance))
    result = []
    if self.managed_by_extended is not None and len(self.managed_by_extended) > 0:
        disks_names = [d.get('name').lower() for p, d in disk_instances]
        attach_config = []
        for vm in managed_vm_id:
            disks = [d for p, d in disk_instances if self._is_disk_attached_to_vm(vm.id, d)]
            if len(disks) > 0:
                attach_config.append(self.create_detachment_configuration(vm, disks_names))
        if len(attach_config) > 0:
            changed = True
            self.update_virtual_machines(attach_config)
        result = self.compute_disks_result(disk_instances)
    elif self.managed_by_extended is None:
        changed = self.detach_disks_from_all_vms(disk_instances)
        if len(disk_instances) > 0:
            disks_ids = [disk.get('id') for param, disk in disk_instances]
            changed = True
            self.delete_disks(disks_ids)
    return dict(changed=changed, state=result)