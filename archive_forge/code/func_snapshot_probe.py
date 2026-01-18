from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def snapshot_probe(self):
    updates = []
    self.rename_validation(updates)
    kwargs = dict(((k, getattr(self, k)) for k in ['old_name', 'parentuid'] if getattr(self, k)))
    ls_data = self.lsvolumegroupsnapshot(**kwargs)
    if self.ownershipgroup and ls_data['owner_name'] != self.ownershipgroup:
        updates.append('ownershipgroup')
    if self.safeguarded in {True, False} and self.safeguarded != strtobool(ls_data.get('safeguarded', 0)):
        self.module.fail_json(msg='Following paramter not applicable for update operation: safeguarded')
    self.log('Snapshot probe result: %s', updates)
    return updates