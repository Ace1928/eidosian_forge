from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def process_operation(self):
    """Calls the create or delete function based on the operation"""
    self.set_operation()
    if self.operation == 'remove':
        results = self.remove_vspan_session()
        self.module.exit_json(**results)
    if self.operation == 'add':
        self.set_port_for_vm()
        results = self.add_vspan_session()
        self.module.exit_json(**results)
    if self.operation == 'edit':
        self.remove_vspan_session()
        self.set_port_for_vm()
        results = self.add_vspan_session()
        self.module.exit_json(**results)