from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import get_all_objs, vmware_argument_spec, find_datacenter_by_name, \
from ansible.module_utils.basic import AnsibleModule
def state_add_rp(self):
    changed = True
    if self.module.check_mode:
        self.module.exit_json(changed=changed)
    rp_spec = self.generate_rp_config()
    if self.parent_resource_pool:
        rootResourcePool = self.compute_resource_obj
    else:
        rootResourcePool = self.compute_resource_obj.resourcePool
    rootResourcePool.CreateResourcePool(self.resource_pool, rp_spec)
    resource_pool_config = self.generate_rp_config_return_value(True)
    self.module.exit_json(changed=changed, resource_pool_config=resource_pool_config)