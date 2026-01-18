from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, find_datacenter_by_name, vmware_argument_spec, wait_for_task
from ansible.module_utils._text import to_native
def destroy_datacenter(self):
    results = dict(changed=False)
    try:
        if self.datacenter_obj and (not self.module.check_mode):
            task = self.datacenter_obj.Destroy_Task()
            changed, result = wait_for_task(task)
            results['changed'] = changed
            results['result'] = result
        self.module.exit_json(**results)
    except (vim.fault.VimFault, vmodl.RuntimeFault, vmodl.MethodFault) as runtime_fault:
        self.module.fail_json(msg="Failed to delete a datacenter '%s' due to : %s" % (self.datacenter_name, to_native(runtime_fault.msg)))
    except Exception as generic_exc:
        self.module.fail_json(msg="Failed to delete a datacenter '%s' due to generic error: %s" % (self.datacenter_name, to_native(generic_exc)))