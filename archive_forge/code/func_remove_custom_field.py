from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
def remove_custom_field(self, vm, user_fields):
    """Remove the value from the existing custom attribute.

        Args:
            vm (vim.VirtualMachine): The managed object of a virtual machine.
            user_fields (list): list of the specified custom attributes by user.

        Returns:
            The dictionary for the ansible return value.
        """
    for v in user_fields:
        v['value'] = ''
    self.check_exists(vm, user_fields)
    if self.module.check_mode is True:
        self.module.exit_json(changed=self.changed, diff=self.diff_config)
    for field in self.update_custom_attributes:
        self.content.customFieldsManager.SetField(entity=vm, key=field['key'], value=field['value'])
        self.result_fields[field['name']] = field['value']
    return {'changed': self.changed, 'failed': False, 'custom_attributes': self.result_fields}