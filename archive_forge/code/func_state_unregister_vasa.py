from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_sms import (
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec
from ansible.module_utils._text import to_native
def state_unregister_vasa(self):
    """
        Unregister VASA provider
        """
    changed, result = (True, None)
    try:
        if not self.module.check_mode:
            uid = self.vasa_provider_info.uid
            task = self.storage_manager.UnregisterProvider_Task(uid)
            changed, result = wait_for_sms_task(task)
        self.module.exit_json(changed=changed, result=result)
    except Exception as generic_exc:
        self.module.fail_json(msg='Failed to unregister VASA due to generic exception %s' % to_native(generic_exc))