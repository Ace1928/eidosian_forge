from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, find_datastore_by_name, find_obj, wait_for_task
from ansible_collections.community.vmware.plugins.module_utils.vmware_sms import SMS
from ansible.module_utils._text import to_native
def check_datastore_host_state(self):
    storage_system = self.esxi.configManager.storageSystem
    host_file_sys_vol_mount_info = storage_system.fileSystemVolumeInfo.mountInfo
    for host_mount_info in host_file_sys_vol_mount_info:
        if host_mount_info.volume.name == self.datastore_name:
            if self.auto_expand and host_mount_info.volume.type == 'VMFS':
                self.expand_datastore_up_to_full()
            return 'present'
    return 'absent'