from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import get_all_objs, vmware_argument_spec, find_datacenter_by_name, \
from ansible.module_utils.basic import AnsibleModule
def generate_rp_config_return_value(self, include_rp_config=False):
    resource_config_return_value = {}
    if include_rp_config:
        resource_config_return_value = self.to_json(self.select_resource_pool().config)
    resource_config_return_value['name'] = self.resource_pool
    return resource_config_return_value