from __future__ import annotations
import sys
import eventlet
def before_fork():
    _prefork_active[0] = _global_dict['_active']
    _global_dict['_active'] = _patched._active