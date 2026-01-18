from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def destroy_dvswitch(self):
    """Delete a DVS"""
    changed = True
    results = dict(changed=changed)
    results['dvswitch'] = self.switch_name
    if self.module.check_mode:
        results['result'] = 'DVS would be deleted'
    else:
        try:
            task = self.dvs.Destroy_Task()
        except vim.fault.VimFault as vim_fault:
            self.module.fail_json(msg='Failed to deleted DVS : %s' % to_native(vim_fault))
        wait_for_task(task)
        results['result'] = 'DVS deleted'
    self.module.exit_json(**results)