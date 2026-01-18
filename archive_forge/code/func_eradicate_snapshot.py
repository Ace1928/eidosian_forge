from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
from datetime import datetime
def eradicate_snapshot(module, blade):
    """Eradicate Snapshot"""
    if not module.check_mode:
        snapname = module.params['name'] + '.' + module.params['suffix']
        try:
            blade.file_system_snapshots.delete_file_system_snapshots(name=snapname)
            changed = True
        except Exception:
            changed = False
    module.exit_json(changed=changed)