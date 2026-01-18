from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def create_fc_partnership(self, restapi, cluster, validate):
    self.create_validation(validate)
    if self.module.check_mode:
        self.changed = True
        return
    cmd = 'mkfcpartnership'
    cmdopts = {'linkbandwidthmbits': self.linkbandwidthmbits}
    if self.backgroundcopyrate:
        cmdopts['backgroundcopyrate'] = self.backgroundcopyrate
    restapi.svc_run_command(cmd, cmdopts, cmdargs=[cluster])
    self.log('FC partnership (%s) created', cluster)
    if self.start:
        restapi.svc_run_command('chpartnership', {'start': True}, [cluster])
        self.log('FC partnership (%s) started', cluster)
    self.changed = True