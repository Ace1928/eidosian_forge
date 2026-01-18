from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, find_datastore_by_name, find_obj, wait_for_task
from ansible_collections.community.vmware.plugins.module_utils.vmware_sms import SMS
from ansible.module_utils._text import to_native
def mount_vmfs_datastore_host(self):
    if self.module.check_mode is False:
        ds_path = '/vmfs/devices/disks/' + str(self.vmfs_device_name)
        host_ds_system = self.esxi.configManager.datastoreSystem
        ds_system = vim.host.DatastoreSystem
        if self.vmfs_device_name in self.get_used_disks_names():
            error_message_used_disk = 'VMFS disk %s already in use' % self.vmfs_device_name
            self.module.fail_json(msg='%s' % error_message_used_disk)
        error_message_mount = 'Cannot mount datastore %s on host %s' % (self.datastore_name, self.esxi.name)
        try:
            if self.resignature:
                storage_system = self.esxi.configManager.storageSystem
                host_unres_volumes = storage_system.QueryUnresolvedVmfsVolume()
                unres_vol_extents = {}
                for unres_vol in host_unres_volumes:
                    for ext in unres_vol.extent:
                        unres_vol_extents[ext.device.diskName] = ext
                if self.vmfs_device_name in unres_vol_extents:
                    spec = vim.host.UnresolvedVmfsResignatureSpec()
                    spec.extentDevicePath = unres_vol_extents[self.vmfs_device_name].devicePath
                    task = host_ds_system.ResignatureUnresolvedVmfsVolume_Task(spec)
                    wait_for_task(task=task)
                    task.info.result.result.RenameDatastore(self.datastore_name)
            else:
                vmfs_ds_options = ds_system.QueryVmfsDatastoreCreateOptions(host_ds_system, ds_path, self.vmfs_version)
                vmfs_ds_options[0].spec.vmfs.volumeName = self.datastore_name
                ds_system.CreateVmfsDatastore(host_ds_system, vmfs_ds_options[0].spec)
        except (vim.fault.NotFound, vim.fault.DuplicateName, vim.fault.HostConfigFault, vmodl.fault.InvalidArgument) as fault:
            self.module.fail_json(msg='%s : %s' % (error_message_mount, to_native(fault.msg)))
        except Exception as e:
            self.module.fail_json(msg='%s : %s' % (error_message_mount, to_native(e)))
    self.module.exit_json(changed=True, result='Datastore %s on host %s' % (self.datastore_name, self.esxi.name))