from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
from datetime import datetime
def recover_snapshot(module, blade):
    """Recover deleted Snapshot"""
    changed = True
    if not module.check_mode:
        snapname = module.params['name'] + '.' + module.params['suffix']
        new_attr = FileSystemSnapshot(destroyed=False)
        try:
            blade.file_system_snapshots.update_file_system_snapshots(name=snapname, attributes=new_attr)
        except Exception:
            changed = False
    module.exit_json(changed=changed)