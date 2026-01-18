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
def set_postmortem_debugger(cls, cmdline, auto=None, hotkey=None, bits=None):
    """
        Sets the postmortem debugging settings in the Registry.

        @warning: This method requires administrative rights.

        @see: L{get_postmortem_debugger}

        @type  cmdline: str
        @param cmdline: Command line to the new postmortem debugger.
            When the debugger is invoked, the first "%ld" is replaced with the
            process ID and the second "%ld" is replaced with the event handle.
            Don't forget to enclose the program filename in double quotes if
            the path contains spaces.

        @type  auto: bool
        @param auto: Set to C{True} if no user interaction is allowed, C{False}
            to prompt a confirmation dialog before attaching.
            Use C{None} to leave this value unchanged.

        @type  hotkey: int
        @param hotkey: Virtual key scan code for the user defined hotkey.
            Use C{0} to disable the hotkey.
            Use C{None} to leave this value unchanged.

        @type  bits: int
        @param bits: Set to C{32} for the 32 bits debugger, or C{64} for the
            64 bits debugger. Set to {None} for the default (L{System.bits}).

        @rtype:  tuple( str, bool, int )
        @return: Previously defined command line and auto flag.

        @raise WindowsError:
            Raises an exception on error.
        """
    if bits is None:
        bits = cls.bits
    elif bits not in (32, 64):
        raise NotImplementedError('Unknown architecture (%r bits)' % bits)
    if bits == 32 and cls.bits == 64:
        keyname = 'HKLM\\SOFTWARE\\Wow6432Node\\Microsoft\\Windows NT\\CurrentVersion\\AeDebug'
    else:
        keyname = 'HKLM\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\AeDebug'
    key = cls.registry[keyname]
    if cmdline is not None:
        key['Debugger'] = cmdline
    if auto is not None:
        key['Auto'] = int(bool(auto))
    if hotkey is not None:
        key['UserDebuggerHotkey'] = int(hotkey)