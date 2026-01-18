from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import get_all_objs, vmware_argument_spec, find_datacenter_by_name, \
from ansible.module_utils.basic import AnsibleModule
def select_resource_pool(self):
    pool_obj = None
    resource_pools = get_all_objs(self.content, [vim.ResourcePool], folder=self.compute_resource_obj)
    pool_selections = self.get_obj([vim.ResourcePool], self.resource_pool, return_all=True)
    if pool_selections:
        for p in pool_selections:
            if p in resource_pools:
                pool_obj = p
                break
    return pool_obj