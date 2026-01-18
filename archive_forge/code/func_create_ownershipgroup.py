from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
def create_ownershipgroup(self):
    if self.module.check_mode:
        self.changed = True
        return
    if self.keepobjects:
        self.module.fail_json(msg='Keepobjects should only be passed while deleting ownershipgroup')
    cmd = 'mkownershipgroup'
    cmdopts = None
    cmdargs = ['-name', self.name]
    result = self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
    self.changed = True
    self.log('Create ownership group result: %s', result)