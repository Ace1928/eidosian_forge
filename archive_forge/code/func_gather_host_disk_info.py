from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
def gather_host_disk_info(self):
    hosts_disk_info = {}
    for host in self.hosts:
        host_disk_info = []
        storage_system = host.configManager.storageSystem.storageDeviceInfo
        lun_lookup = {}
        for lun in storage_system.multipathInfo.lun:
            key = lun.lun
            paths = []
            for path in lun.path:
                paths.append(path.name)
            lun_lookup[key] = paths
        for disk in storage_system.scsiLun:
            canonical_name = disk.canonicalName
            try:
                capacity = int(disk.capacity.block * disk.capacity.blockSize / 1048576)
            except AttributeError:
                capacity = 0
            try:
                device_path = disk.devicePath
            except AttributeError:
                device_path = ''
            device_type = disk.deviceType
            display_name = disk.displayName
            disk_uid = disk.key
            device_ctd_list = lun_lookup[disk_uid]
            disk_dict = {'capacity_mb': capacity, 'device_path': device_path, 'device_type': device_type, 'display_name': display_name, 'disk_uid': disk_uid, 'device_ctd_list': device_ctd_list, 'canonical_name': canonical_name}
            host_disk_info.append(disk_dict)
        hosts_disk_info[host.name] = host_disk_info
    return hosts_disk_info