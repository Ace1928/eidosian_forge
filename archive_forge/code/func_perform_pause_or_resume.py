from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def perform_pause_or_resume(self, pause, replication_pair_details, pair_id):
    changed = False
    if pause and replication_pair_details['initialCopyState'] not in ('Paused', 'Done'):
        changed = self.pause(pair_id)
    elif not pause and replication_pair_details['initialCopyState'] == 'Paused':
        changed = self.resume(pair_id)
    return changed