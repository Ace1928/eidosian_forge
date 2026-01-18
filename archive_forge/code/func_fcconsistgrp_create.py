from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def fcconsistgrp_create(self):
    if self.module.check_mode:
        self.changed = True
        return
    cmd = 'mkfcconsistgrp'
    cmdopts = {}
    cmdopts['name'] = self.name
    if self.ownershipgroup:
        cmdopts['ownershipgroup'] = self.ownershipgroup
    self.log('Creating fc consistgrp.. Command: %s opts %s', cmd, cmdopts)
    result = self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
    if 'message' in result:
        self.changed = True
        self.log('Create fc consistgrp message %s', result['message'])
    else:
        self.module.fail_json(msg='Failed to create fc consistgrp [%s]' % self.name)