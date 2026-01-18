import glob
import os
import struct
from .lazy_import import lazy_import
import ctypes
from breezy import cmdline
from breezy.i18n import gettext
def set_file_attr_hidden(path):
    """Set file attributes to hidden if possible"""
    from ctypes.wintypes import BOOL, DWORD, LPWSTR
    SetFileAttributes = ctypes.windll.kernel32.SetFileAttributesW
    SetFileAttributes.argtypes = (LPWSTR, DWORD)
    SetFileAttributes.restype = BOOL
    FILE_ATTRIBUTE_HIDDEN = 2
    if not SetFileAttributes(path, FILE_ATTRIBUTE_HIDDEN):
        e = ctypes.WinError()
        from . import trace
        trace.mutter('Unable to set hidden attribute on %r: %s', path, e)