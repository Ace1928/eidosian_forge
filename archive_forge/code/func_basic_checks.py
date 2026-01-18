from __future__ import absolute_import, division, print_function
from traceback import format_exc
import json
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_ssh import IBMSVCssh
from ansible.module_utils._text import to_native
def basic_checks(self):
    if self.state == 'present':
        if not self.remote_clustername:
            self.module.fail_json(msg='Missing mandatory parameter: remote_clustername')
        if not self.remote_username:
            self.module.fail_json(msg='Missing mandatory parameter: remote_username')
        if not self.remote_password:
            self.module.fail_json(msg='Missing mandatory parameter: remote_password')
    elif self.state == 'absent':
        if not self.remote_clustername:
            self.module.fail_json(msg='Missing mandatory parameter: remote_clustername')
        unsupported = ('remote_username', 'remote_password')
        unsupported_exists = ', '.join((field for field in unsupported if getattr(self, field)))
        if unsupported_exists:
            self.module.fail_json(msg='state=absent but following paramters have been passed: {0}'.format(unsupported_exists))