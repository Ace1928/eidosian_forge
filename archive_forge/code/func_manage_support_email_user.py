from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
import time
def manage_support_email_user(self):
    if self.module.check_mode:
        self.changed = True
        return
    support_email = {}
    selected_email_id = ''
    t = -1 * (time.timezone / 60 / 60)
    if t >= -8 and t <= -4:
        selected_email_id = 'callhome0@de.ibm.com'
    else:
        selected_email_id = 'callhome1@de.ibm.com'
    existing_user = self.restapi.svc_obj_info('lsemailuser', cmdopts=None, cmdargs=None)
    if existing_user:
        for user in existing_user:
            if user['user_type'] == 'support':
                support_email = user
    if not support_email:
        self.log("Creating support email user '%s'.", selected_email_id)
        command = 'mkemailuser'
        command_options = {'address': selected_email_id, 'usertype': 'support', 'info': 'off', 'warning': 'off'}
        if self.inventory:
            command_options['inventory'] = self.inventory
        cmdargs = None
        result = self.restapi.svc_run_command(command, command_options, cmdargs)
        if 'message' in result:
            self.changed = True
            self.log("create support email user result message '%s'", result['message'])
        else:
            self.module.fail_json(msg='Failed to support create email user [%s]' % self.contact_email)
    else:
        modify = {}
        if support_email['address'] != selected_email_id:
            modify['address'] = selected_email_id
        if self.inventory:
            if support_email['inventory'] != self.inventory:
                modify['inventory'] = self.inventory
        if modify:
            self.restapi.svc_run_command('chemailuser', modify, [support_email['id']])
            self.log('Updated support user successfully.')