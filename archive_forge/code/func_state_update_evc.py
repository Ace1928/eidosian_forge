from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def state_update_evc(self):
    """
        Update EVC Mode
        """
    changed, result = (False, None)
    try:
        if not self.module.check_mode and self.current_evc_mode != self.evc_mode:
            evc_task = self.evcm.ConfigureEvcMode_Task(self.evc_mode)
            changed, result = wait_for_task(evc_task)
        if self.module.check_mode and self.current_evc_mode != self.evc_mode:
            changed = True
        if self.current_evc_mode == self.evc_mode:
            self.module.exit_json(changed=changed, msg="EVC Mode is already set to '%(evc_mode)s' on '%(cluster_name)s'." % self.params)
        self.module.exit_json(changed=changed, msg="EVC Mode has been updated to '%(evc_mode)s' on '%(cluster_name)s'." % self.params)
    except TaskError as invalid_argument:
        self.module.fail_json(msg='Failed to update EVC mode: %s' % to_native(invalid_argument))