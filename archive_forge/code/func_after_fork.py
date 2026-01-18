from __future__ import nested_scopes
import platform
import weakref
import struct
import warnings
import functools
from contextlib import contextmanager
import sys  # Note: the sys import must be here anyways (others depend on it)
import codecs as _codecs
import os
from _pydevd_bundle import pydevd_vm_type
from _pydev_bundle._pydev_saved_modules import thread, threading
def after_fork():
    """
    Must be called after a fork operation (will reset the ForkSafeLock).
    """
    global _fork_safe_locks
    locks = _fork_safe_locks[:]
    _fork_safe_locks = []
    for lock in locks:
        lock = lock()
        if lock is not None:
            lock._init()