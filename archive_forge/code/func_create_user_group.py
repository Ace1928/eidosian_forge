from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def create_user_group(self):
    if self.noownershipgroup:
        self.module.fail_json(msg='Parameter [noownershipgroup] is not applicable while creating a usergroup')
    if not self.role:
        self.module.fail_json(msg='Missing mandatory parameter: role')
    if self.module.check_mode:
        self.changed = True
        return
    command = 'mkusergrp'
    command_options = {'name': self.name}
    if self.role:
        command_options['role'] = self.role
    if self.ownershipgroup:
        command_options['ownershipgroup'] = self.ownershipgroup
    result = self.restapi.svc_run_command(command, command_options, cmdargs=None)
    self.log('create user group result %s', result)
    if 'message' in result:
        self.changed = True
        self.log('create user group result message %s', result['message'])
    else:
        self.module.fail_json(msg='Failed to user volume group [%s]' % self.name)