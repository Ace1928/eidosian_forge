from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
import time
def create_email_server(self):
    if self.module.check_mode:
        self.changed = True
        return
    self.log("Creating email server '%s:%s'.", self.serverIP, self.serverPort)
    command = 'mkemailserver'
    command_options = {'ip': self.serverIP, 'port': self.serverPort}
    cmdargs = None
    result = self.restapi.svc_run_command(command, command_options, cmdargs)
    if 'message' in result:
        self.changed = True
        self.log("create email server result message '%s'", result['message'])
    else:
        self.module.fail_json(msg='Failed to create email server [%s:%s]' % (self.serverIP, self.serverPort))