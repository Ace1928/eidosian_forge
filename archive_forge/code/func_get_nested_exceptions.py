from winappdbg import win32
from winappdbg import compat
from winappdbg.win32 import FileHandle, ProcessHandle, ThreadHandle
from winappdbg.breakpoint import ApiHook
from winappdbg.module import Module
from winappdbg.thread import Thread
from winappdbg.process import Process
from winappdbg.textio import HexDump
from winappdbg.util import StaticClass, PathOperations
import sys
import ctypes
import warnings
import traceback
def get_nested_exceptions(self):
    """
        Traverses the exception record linked list and builds a Python list.

        Nested exception records are received for nested exceptions. This
        happens when an exception is raised in the debugee while trying to
        handle a previous exception.

        @rtype:  list( L{ExceptionEvent} )
        @return:
            List of ExceptionEvent objects representing each exception record
            found in this event.

            There is always at least one exception record, so the list is
            never empty. All other methods of this class read from the first
            exception record only, that is, the most recent exception.
        """
    nested = [self]
    raw = self.raw
    dwDebugEventCode = raw.dwDebugEventCode
    dwProcessId = raw.dwProcessId
    dwThreadId = raw.dwThreadId
    dwFirstChance = raw.u.Exception.dwFirstChance
    record = raw.u.Exception.ExceptionRecord
    while True:
        record = record.ExceptionRecord
        if not record:
            break
        raw = win32.DEBUG_EVENT()
        raw.dwDebugEventCode = dwDebugEventCode
        raw.dwProcessId = dwProcessId
        raw.dwThreadId = dwThreadId
        raw.u.Exception.ExceptionRecord = record
        raw.u.Exception.dwFirstChance = dwFirstChance
        event = EventFactory.get(self.debug, raw)
        nested.append(event)
    return nested