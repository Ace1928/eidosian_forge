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
def call_only_once(func):
    """
    To be used as a decorator

    @call_only_once
    def func():
        print 'Calling func only this time'

    Actually, in PyDev it must be called as:

    func = call_only_once(func) to support older versions of Python.
    """

    def new_func(*args, **kwargs):
        if not new_func._called:
            new_func._called = True
            return func(*args, **kwargs)
    new_func._called = False
    return new_func