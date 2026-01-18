from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils._text import to_native
def state_destroy_cluster(self):
    """
        Destroy cluster
        """
    changed, result = (True, None)
    try:
        if not self.module.check_mode:
            task = self.cluster.Destroy_Task()
            changed, result = wait_for_task(task)
        self.module.exit_json(changed=changed, result=result)
    except vim.fault.VimFault as vim_fault:
        self.module.fail_json(msg=to_native(vim_fault.msg))
    except vmodl.RuntimeFault as runtime_fault:
        self.module.fail_json(msg=to_native(runtime_fault.msg))
    except vmodl.MethodFault as method_fault:
        self.module.fail_json(msg=to_native(method_fault.msg))
    except Exception as generic_exc:
        self.module.fail_json(msg='Failed to destroy cluster due to generic exception %s' % to_native(generic_exc))