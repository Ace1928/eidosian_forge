from __future__ import absolute_import, division, print_function
import traceback
from random import randint
from ansible.module_utils.common.network import is_mac
from ansible.module_utils.basic import missing_required_lib
def gather_disk_info(self, vm_obj):
    """
        Gather information about VM's disks
        Args:
            vm_obj: Managed object of virtual machine
        Returns: A list of dict containing disks information
        """
    controller_info = dict()
    disks_info = dict()
    if vm_obj is None:
        return disks_info
    controller_index = 0
    for controller in vm_obj.config.hardware.device:
        for name, type in self.disk_ctl_device_type.items():
            if isinstance(controller, type):
                controller_info[controller_index] = dict(key=controller.key, controller_type=name, bus_number=controller.busNumber, devices=controller.device)
                controller_index += 1
    disk_index = 0
    for disk in vm_obj.config.hardware.device:
        if isinstance(disk, vim.vm.device.VirtualDisk):
            if disk.storageIOAllocation is None:
                disk.storageIOAllocation = vim.StorageResourceManager.IOAllocationInfo()
                disk.storageIOAllocation.shares = vim.SharesInfo()
            if disk.shares is None:
                disk.shares = vim.SharesInfo()
            disks_info[disk_index] = dict(key=disk.key, label=disk.deviceInfo.label, summary=disk.deviceInfo.summary, backing_filename=disk.backing.fileName, backing_datastore=disk.backing.datastore.name, backing_sharing=disk.backing.sharing if hasattr(disk.backing, 'sharing') else None, backing_uuid=disk.backing.uuid if hasattr(disk.backing, 'uuid') else None, backing_writethrough=disk.backing.writeThrough if hasattr(disk.backing, 'writeThrough') else None, backing_diskmode=disk.backing.diskMode if hasattr(disk.backing, 'diskMode') else None, backing_disk_mode=disk.backing.diskMode if hasattr(disk.backing, 'diskMode') else None, iolimit_limit=disk.storageIOAllocation.limit, iolimit_shares_level=disk.storageIOAllocation.shares.level, iolimit_shares_limit=disk.storageIOAllocation.shares.shares, shares_level=disk.shares.level, shares_limit=disk.shares.shares, controller_key=disk.controllerKey, unit_number=disk.unitNumber, capacity_in_kb=disk.capacityInKB, capacity_in_bytes=disk.capacityInBytes)
            if isinstance(disk.backing, vim.vm.device.VirtualDisk.FlatVer1BackingInfo):
                disks_info[disk_index]['backing_type'] = 'FlatVer1'
            elif isinstance(disk.backing, vim.vm.device.VirtualDisk.FlatVer2BackingInfo):
                disks_info[disk_index]['backing_type'] = 'FlatVer2'
                disks_info[disk_index]['backing_thinprovisioned'] = disk.backing.thinProvisioned
                disks_info[disk_index]['backing_eagerlyscrub'] = disk.backing.eagerlyScrub
            elif isinstance(disk.backing, vim.vm.device.VirtualDisk.LocalPMemBackingInfo):
                disks_info[disk_index]['backing_type'] = 'LocalPMem'
                disks_info[disk_index]['backing_volumeuuid'] = disk.backing.volumeUUID
            elif isinstance(disk.backing, vim.vm.device.VirtualDisk.PartitionedRawDiskVer2BackingInfo):
                disks_info[disk_index]['backing_type'] = 'PartitionedRawDiskVer2'
                disks_info[disk_index]['backing_descriptorfilename'] = disk.backing.descriptorFileName
            elif isinstance(disk.backing, vim.vm.device.VirtualDisk.RawDiskMappingVer1BackingInfo):
                disks_info[disk_index]['backing_type'] = 'RawDiskMappingVer1'
                disks_info[disk_index]['backing_devicename'] = disk.backing.deviceName
                disks_info[disk_index]['backing_lunuuid'] = disk.backing.lunUuid
                disks_info[disk_index]['backing_compatibility_mode'] = disk.backing.compatibilityMode
            elif isinstance(disk.backing, vim.vm.device.VirtualDisk.RawDiskVer2BackingInfo):
                disks_info[disk_index]['backing_type'] = 'RawDiskVer2'
                disks_info[disk_index]['backing_descriptorfilename'] = disk.backing.descriptorFileName
            elif isinstance(disk.backing, vim.vm.device.VirtualDisk.SeSparseBackingInfo):
                disks_info[disk_index]['backing_type'] = 'SeSparse'
            elif isinstance(disk.backing, vim.vm.device.VirtualDisk.SparseVer1BackingInfo):
                disks_info[disk_index]['backing_type'] = 'SparseVer1'
                disks_info[disk_index]['backing_spaceusedinkb'] = disk.backing.spaceUsedInKB
                disks_info[disk_index]['backing_split'] = disk.backing.split
            elif isinstance(disk.backing, vim.vm.device.VirtualDisk.SparseVer2BackingInfo):
                disks_info[disk_index]['backing_type'] = 'SparseVer2'
                disks_info[disk_index]['backing_spaceusedinkb'] = disk.backing.spaceUsedInKB
                disks_info[disk_index]['backing_split'] = disk.backing.split
            for controller_index in range(len(controller_info)):
                if controller_info[controller_index]['key'] == disks_info[disk_index]['controller_key']:
                    disks_info[disk_index]['controller_bus_number'] = controller_info[controller_index]['bus_number']
                    disks_info[disk_index]['controller_type'] = controller_info[controller_index]['controller_type']
            disk_index += 1
    return disks_info