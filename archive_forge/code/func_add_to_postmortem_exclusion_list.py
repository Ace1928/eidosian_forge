from __future__ import with_statement
from winappdbg import win32
from winappdbg.registry import Registry
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import Regenerator, PathOperations, MemoryAddresses, DebugRegister, \
from winappdbg.process import _ProcessContainer
from winappdbg.window import Window
import sys
import os
import ctypes
import warnings
from os import path, getenv
@classmethod
def add_to_postmortem_exclusion_list(cls, pathname, bits=None):
    """
        Adds the given filename to the exclusion list for postmortem debugging.

        @warning: This method requires administrative rights.

        @see: L{get_postmortem_exclusion_list}

        @type  pathname: str
        @param pathname:
            Application pathname to exclude from postmortem debugging.

        @type  bits: int
        @param bits: Set to C{32} for the 32 bits debugger, or C{64} for the
            64 bits debugger. Set to {None} for the default (L{System.bits}).

        @raise WindowsError:
            Raises an exception on error.
        """
    if bits is None:
        bits = cls.bits
    elif bits not in (32, 64):
        raise NotImplementedError('Unknown architecture (%r bits)' % bits)
    if bits == 32 and cls.bits == 64:
        keyname = 'HKLM\\SOFTWARE\\Wow6432Node\\Microsoft\\Windows NT\\CurrentVersion\\AeDebug\\AutoExclusionList'
    else:
        keyname = 'HKLM\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\AeDebug\\AutoExclusionList'
    try:
        key = cls.registry[keyname]
    except KeyError:
        key = cls.registry.create(keyname)
    key[pathname] = 1