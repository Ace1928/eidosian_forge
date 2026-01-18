from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
import time
def create_email_user(self):
    if self.module.check_mode:
        self.changed = True
        return
    self.log("Creating email user '%s'.", self.contact_email)
    command = 'mkemailuser'
    command_options = {'address': self.contact_email, 'usertype': 'local'}
    if self.inventory:
        command_options['inventory'] = self.inventory
    cmdargs = None
    result = self.restapi.svc_run_command(command, command_options, cmdargs)
    if 'message' in result:
        self.changed = True
        self.log("Create email user result message '%s'.", result['message'])
    else:
        self.module.fail_json(msg='Failed to create email user [%s]' % self.contact_email)