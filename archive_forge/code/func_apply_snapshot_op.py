from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, list_snapshots, vmware_argument_spec
def apply_snapshot_op(self, vm):
    result = {}
    if self.module.params['state'] == 'present':
        if self.module.params['new_snapshot_name'] or self.module.params['new_description']:
            self.rename_snapshot(vm)
            result = {'changed': True, 'failed': False, 'renamed': True}
            task = None
        else:
            task = self.snapshot_vm(vm)
    elif self.module.params['state'] in ['absent', 'revert']:
        task = self.remove_or_revert_snapshot(vm)
    elif self.module.params['state'] == 'remove_all':
        task = vm.RemoveAllSnapshots()
    else:
        raise AssertionError()
    if task:
        self.wait_for_task(task)
        if task.info.state == 'error':
            result = {'changed': False, 'failed': True, 'msg': task.info.error.msg}
        else:
            result = {'changed': True, 'failed': False, 'snapshot_results': list_snapshots(vm)}
    return result