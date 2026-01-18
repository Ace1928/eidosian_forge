from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import get_all_objs, vmware_argument_spec, find_datacenter_by_name, \
from ansible.module_utils.basic import AnsibleModule
def check_rp_state(self):
    self.resource_pool_obj = self.select_resource_pool()
    if self.resource_pool_obj is None:
        return 'absent'
    return 'present'