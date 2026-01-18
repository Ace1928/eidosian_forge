from __future__ import annotations
import sys
import eventlet
def fix_threading_active(_global_dict=_threading.current_thread.__globals__, _patched=orig_mod):
    _prefork_active = [None]

    def before_fork():
        _prefork_active[0] = _global_dict['_active']
        _global_dict['_active'] = _patched._active

    def after_fork():
        _global_dict['_active'] = _prefork_active[0]
    register_at_fork(before=before_fork, after_in_parent=after_fork)