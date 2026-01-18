from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def license_update(self, modify):
    if self.module.check_mode:
        self.changed = True
        return
    cmd = 'chlicense'
    for license in modify:
        cmdopts = {}
        cmdopts[license] = getattr(self, license)
        self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
    self.changed = True if modify else False
    if self.encryption:
        cmdopts = {}
        cmdopts['encryption'] = self.encryption
        self.changed = True
        self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
    self.log('Licensed functions %s updated', modify)
    self.message += ' Licensed functions %s updated.' % modify