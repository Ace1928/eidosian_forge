from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def fcconsistgrp_update(self, modify):
    if self.module.check_mode:
        self.changed = True
        return
    if modify:
        self.log('updating fcmap with properties %s', modify)
        cmd = 'chfcconsistgrp'
        cmdopts = {}
        for prop in modify:
            cmdopts[prop] = modify[prop]
        cmdargs = [self.name]
        self.restapi.svc_run_command(cmd, cmdopts, cmdargs)