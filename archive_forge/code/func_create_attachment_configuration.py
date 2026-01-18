from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_attachment_configuration(self, vm, disks):
    vm_id = parse_resource_id(vm.id)
    for managed_disk, disk_instance in disks:
        lun = managed_disk.get('lun')
        if lun is None:
            luns = [d.lun for d in vm.storage_profile.data_disks] if vm.storage_profile.data_disks else []
            lun = 0
            while True:
                if lun not in luns:
                    break
                lun = lun + 1
            for item in vm.storage_profile.data_disks:
                if item.name == managed_disk.get('name'):
                    lun = item.lun
        params = self.compute_models.ManagedDiskParameters(id=disk_instance.get('id'), storage_account_type=disk_instance.get('storage_account_type'))
        attach_caching = managed_disk.get('attach_caching')
        caching_options = self.compute_models.CachingTypes[attach_caching] if attach_caching and attach_caching != '' else None
        data_disk = self.compute_models.DataDisk(lun=lun, create_option=self.compute_models.DiskCreateOptionTypes.attach, managed_disk=params, caching=caching_options)
        vm.storage_profile.data_disks.append(data_disk)
    return (vm_id['resource_group'], vm_id['resource_name'], vm)