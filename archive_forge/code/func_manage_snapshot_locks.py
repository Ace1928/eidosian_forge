from __future__ import (absolute_import, division, print_function)
from functools import wraps
from os import environ
from os import path
from datetime import datetime
def manage_snapshot_locks(module, snapshot):
    """
    Manage the locking of a snapshot. Check for bad lock times.
    See check_snapshot_lock_options() which has additional checks.
    """
    snapshot_lock_expires_at = module.params['snapshot_lock_expires_at']
    snap_is_locked = snapshot.get_lock_state() == 'LOCKED'
    current_lock_expires_at = snapshot.get_lock_expires_at()
    changed = False
    check_snapshot_lock_options(module)
    if snapshot_lock_expires_at:
        lock_expires_at = arrow.get(snapshot_lock_expires_at)
        if snap_is_locked and lock_expires_at < current_lock_expires_at:
            msg = f"snapshot_lock_expires_at '{lock_expires_at}' preceeds the current lock time of '{current_lock_expires_at}'"
            module.fail_json(msg=msg)
        elif snap_is_locked and lock_expires_at == current_lock_expires_at:
            pass
        else:
            if not module.check_mode:
                snapshot.update_lock_expires_at(lock_expires_at)
            changed = True
    return changed