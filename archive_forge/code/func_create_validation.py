from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import \
from ansible.module_utils._text import to_native
def create_validation(self):
    mutually_exclusive = (('ownershipgroup', 'safeguardpolicyname'), ('ownershipgroup', 'snapshotpolicy'), ('ownershipgroup', 'policystarttime'), ('snapshotpolicy', 'safeguardpolicyname'), ('replicationpolicy', 'noreplicationpolicy'))
    for param1, param2 in mutually_exclusive:
        if getattr(self, param1) and getattr(self, param2):
            self.module.fail_json(msg='Mutually exclusive parameters: {0}, {1}'.format(param1, param2))
    unsupported = ('nosafeguardpolicy', 'noownershipgroup', 'nosnapshotpolicy', 'snapshotpolicysuspended', 'noreplicationpolicy')
    unsupported_exists = ', '.join((field for field in unsupported if getattr(self, field)))
    if unsupported_exists:
        self.module.fail_json(msg='Following paramters not supported during creation scenario: {0}'.format(unsupported_exists))
    if self.type and (not self.snapshot):
        self.module.fail_json(msg='type={0} but following parameter is missing: snapshot'.format(self.type))