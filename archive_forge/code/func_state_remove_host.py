from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def state_remove_host(self):
    """Remove host from vCenter"""
    changed = True
    result = None
    if self.module.check_mode:
        result = "Host would be removed from vCenter '%s'" % self.vcenter
    else:
        parent_type = self.get_parent_type(self.host_update)
        if parent_type == 'cluster':
            self.put_host_in_maintenance_mode(self.host_update)
        try:
            if self.folder_name:
                task = self.host_parent_compute_resource.Destroy_Task()
            elif self.cluster_name:
                task = self.host.Destroy_Task()
        except vim.fault.VimFault as vim_fault:
            self.module.fail_json(msg=vim_fault)
        try:
            changed, result = wait_for_task(task)
            result = "Host removed from vCenter '%s'" % self.vcenter
        except TaskError as task_error:
            self.module.fail_json(msg="Failed to remove the host from vCenter '%s' : %s" % (self.vcenter, to_native(task_error)))
    self.module.exit_json(changed=changed, result=str(result))