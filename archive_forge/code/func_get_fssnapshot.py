from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
from datetime import datetime
def get_fssnapshot(module, blade):
    """Return Snapshot or None"""
    try:
        filt = "source='" + module.params['name'] + "' and suffix='" + module.params['suffix'] + "'"
        res = blade.file_system_snapshots.list_file_system_snapshots(filter=filt)
        return res.items[0]
    except Exception:
        return None