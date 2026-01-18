from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def handle_main_entity(self):
    """
        Handles the Ansible task
        """
    if self.command and self.command == 'find':
        self._handle_find()
    elif self.command and self.command == 'change_password':
        self._handle_change_password()
    elif self.command and self.command == 'wait_for_job':
        self._handle_wait_for_job()
    elif self.command and self.command == 'get_csp_enterprise':
        self._handle_get_csp_enterprise()
    elif self.state == 'present':
        self._handle_present()
    elif self.state == 'absent':
        self._handle_absent()
    self.module.exit_json(**self.result)