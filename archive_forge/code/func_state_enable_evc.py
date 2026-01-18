from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def state_enable_evc(self):
    """
        Enable EVC Mode
        """
    changed, result = (False, None)
    try:
        if not self.module.check_mode:
            evc_task = self.evcm.ConfigureEvcMode_Task(self.evc_mode)
            changed, result = wait_for_task(evc_task)
        if self.module.check_mode:
            changed = True
        self.module.exit_json(changed=changed, msg="EVC Mode for '%(evc_mode)s' has been enabled on '%(cluster_name)s'." % self.params)
    except TaskError as invalid_argument:
        self.module.fail_json(msg='Failed to enable EVC mode: %s' % to_native(invalid_argument))