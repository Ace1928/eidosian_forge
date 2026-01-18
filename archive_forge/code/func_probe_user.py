from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def probe_user(self, data):
    properties = {}
    if self.usergroup:
        if self.usergroup != data['usergrp_name']:
            properties['usergrp'] = self.usergroup
    if self.user_password:
        properties['password'] = self.user_password
    if self.nopassword:
        if data['password'] == 'yes':
            properties['nopassword'] = True
    if self.keyfile:
        properties['keyfile'] = self.keyfile
    if self.nokey:
        if data['ssh_key'] == 'yes':
            properties['nokey'] = True
    if self.lock:
        properties['lock'] = True
    if self.unlock:
        properties['unlock'] = True
    if self.forcepasswordchange:
        properties['forcepasswordchange'] = True
    return properties