from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, \
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
def get_tag_info(self, vm_dynamic_obj):
    return self.vmware_client.get_tags_for_vm(vm_mid=vm_dynamic_obj._moId)