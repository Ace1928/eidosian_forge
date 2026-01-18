from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, find_datastore_by_name, find_obj, wait_for_task
from ansible_collections.community.vmware.plugins.module_utils.vmware_sms import SMS
from ansible.module_utils._text import to_native
def mount_nfs_datastore_host(self):
    if self.module.check_mode is False:
        mnt_specs = vim.host.NasVolume.Specification()
        if self.datastore_type == 'nfs':
            mnt_specs.type = 'NFS'
            mnt_specs.remoteHost = self.nfs_server
        if self.datastore_type == 'nfs41':
            mnt_specs.type = 'NFS41'
            mnt_specs.remoteHost = 'something'
            mnt_specs.remoteHostNames = [self.nfs_server]
        mnt_specs.remotePath = self.nfs_path
        mnt_specs.localPath = self.datastore_name
        if self.nfs_ro:
            mnt_specs.accessMode = 'readOnly'
        else:
            mnt_specs.accessMode = 'readWrite'
        error_message_mount = 'Cannot mount datastore %s on host %s' % (self.datastore_name, self.esxi.name)
        try:
            ds = self.esxi.configManager.datastoreSystem.CreateNasDatastore(mnt_specs)
            if not ds:
                self.module.fail_json(msg=error_message_mount)
        except (vim.fault.NotFound, vim.fault.DuplicateName, vim.fault.AlreadyExists, vim.fault.HostConfigFault, vmodl.fault.InvalidArgument, vim.fault.NoVirtualNic, vim.fault.NoGateway) as fault:
            self.module.fail_json(msg='%s: %s' % (error_message_mount, to_native(fault.msg)))
        except Exception as e:
            self.module.fail_json(msg='%s : %s' % (error_message_mount, to_native(e)))
    self.module.exit_json(changed=True, result='Datastore %s on host %s' % (self.datastore_name, self.esxi.name))