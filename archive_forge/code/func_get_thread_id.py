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
def get_thread_id(thread):
    try:
        tid = thread.__pydevd_id__
        if tid is None:
            raise AttributeError()
    except AttributeError:
        tid = _get_or_compute_thread_id_with_lock(thread, is_current_thread=False)
    return tid