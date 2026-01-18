from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
def check_datastore_host(self, esxi_host, datastore):
    """ Get all datastores of specified ESXi host """
    esxi = self.find_hostsystem_by_name(esxi_host)
    if esxi is None:
        self.module.fail_json(msg='Failed to find ESXi hostname %s ' % esxi_host)
    storage_system = esxi.configManager.storageSystem
    host_file_sys_vol_mount_info = storage_system.fileSystemVolumeInfo.mountInfo
    for host_mount_info in host_file_sys_vol_mount_info:
        if host_mount_info.volume.name == datastore:
            return host_mount_info
    return None