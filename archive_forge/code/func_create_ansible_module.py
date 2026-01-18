from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def create_ansible_module(self, **kwargs):
    return self.create_ansible_module_helper(AnsibleModule, (), **kwargs)