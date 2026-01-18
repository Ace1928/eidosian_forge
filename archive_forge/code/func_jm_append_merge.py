import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
def jm_append_merge(dev):
    """
    Append current path to list or merge into last path of the list.
    (1) Append if first path, different item lists or not a 'stroke' version
        of previous path
    (2) If new path has the same items, merge its content into previous path
        and change path["type"] to "fs".
    (3) If "out" is callable, skip the previous and pass dictionary to it.
    """
    assert isinstance(dev.out, list)
    if callable(dev.method) or dev.method:
        if dev.method is None:
            assert 0
        else:
            resp = getattr(dev.out, dev.method)(dev.pathdict)
        if not resp:
            message('calling cdrawings callback function/method failed!', file=sys.stderr)
        dev.pathdict = None
        return

    def append():
        dev.out.append(dev.pathdict.copy())
        dev.pathdict.clear()
    assert isinstance(dev.out, list)
    len_ = len(dev.out)
    if len_ == 0:
        return append()
    thistype = dev.pathdict[dictkey_type]
    if thistype != 's':
        return append()
    prev = dev.out[len_ - 1]
    prevtype = prev[dictkey_type]
    if prevtype != 'f':
        return append()
    previtems = prev[dictkey_items]
    thisitems = dev.pathdict[dictkey_items]
    if previtems != thisitems:
        return append()
    try:
        for k, v in dev.pathdict.items():
            if k not in prev:
                prev[k] = v
        rc = 0
    except Exception:
        if g_exceptions_verbose:
            exception_info()
        rc = -1
    if rc == 0:
        prev[dictkey_type] = 'fs'
        dev.pathdict.clear()
    else:
        message('could not merge stroke and fill path', file=sys.stderr)
        append()