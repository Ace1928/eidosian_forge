from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import get_all_objs, vmware_argument_spec, find_datacenter_by_name, \
from ansible.module_utils.basic import AnsibleModule
def state_remove_rp(self):
    changed = True
    result = None
    if self.module.check_mode:
        self.module.exit_json(changed=changed)
    resource_pool_config = self.generate_rp_config_return_value(True)
    try:
        task = self.resource_pool_obj.Destroy()
        success, result = wait_for_task(task)
    except Exception:
        self.module.fail_json(msg="Failed to remove resource pool '%s' '%s'" % (self.resource_pool, self.resource_pool))
    self.module.exit_json(changed=changed, resource_pool_config=resource_pool_config)