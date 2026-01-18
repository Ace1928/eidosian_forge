from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def disable_sra(self):
    if self.module.check_mode:
        self.changed = True
        return
    cmd = 'chsra'
    cmdopts = {}
    if self.support == 'remote':
        cmdargs = ['-remotesupport', 'disable']
        self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
    cmdargs = ['-disable']
    self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
    self.log('%s support assistance disabled', self.support.capitalize())
    self.remove_proxy_details()
    self.changed = True