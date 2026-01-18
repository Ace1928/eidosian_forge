from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def create_ansible_module_helper(self, clazz, args, **kwargs):
    return clazz(*args, argument_spec=self.argument_spec, mutually_exclusive=self.mutually_exclusive, required_together=self.required_together, required_one_of=self.required_one_of, required_if=self.required_if, required_by=self.required_by, **kwargs)