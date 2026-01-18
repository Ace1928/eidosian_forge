from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, find_datastore_by_name, find_obj, wait_for_task
from ansible_collections.community.vmware.plugins.module_utils.vmware_sms import SMS
from ansible.module_utils._text import to_native
def get_used_disks_names(self):
    used_disks = []
    storage_system = self.esxi.configManager.storageSystem
    for each_vol_mount_info in storage_system.fileSystemVolumeInfo.mountInfo:
        if hasattr(each_vol_mount_info.volume, 'extent'):
            for each_partition in each_vol_mount_info.volume.extent:
                used_disks.append(each_partition.diskName)
    return used_disks