from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def snapshot_policy_probe(self):
    field_mappings = (('backupinterval', self.snapshot_policy_details['backup_interval']), ('backupstarttime', self.snapshot_policy_details['backup_start_time']), ('retentiondays', self.snapshot_policy_details['retention_days']), ('backupunit', self.snapshot_policy_details['backup_unit']))
    updates = []
    for field, existing_value in field_mappings:
        if field == 'backupstarttime':
            updates.append(existing_value != '{0}00'.format(getattr(self, field)))
        else:
            updates.append(existing_value != getattr(self, field))
    return updates