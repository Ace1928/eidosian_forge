from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def create_aws_account(self):
    self.create_validation()
    if self.module.check_mode:
        self.changed = True
        return
    cmd = 'mkcloudaccountawss3'
    cmdopts = {'name': self.name, 'bucketprefix': self.bucketprefix, 'accesskeyid': self.accesskeyid, 'secretaccesskey': self.secretaccesskey}
    params = {'upbandwidthmbits', 'downbandwidthmbits', 'region', 'encrypt'}
    cmdopts.update(dict(((key, getattr(self, key)) for key in params if getattr(self, key))))
    self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None, timeout=20)
    self.log('Cloud account (%s) created', self.name)
    self.changed = True