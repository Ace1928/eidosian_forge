from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump, HexInput
from winappdbg.util import Regenerator, PathOperations, MemoryAddresses
from winappdbg.module import Module, _ModuleContainer
from winappdbg.thread import Thread, _ThreadContainer
from winappdbg.window import Window
from winappdbg.search import Search, \
from winappdbg.disasm import Disassembler
import re
import os
import os.path
import ctypes
import struct
import warnings
import traceback
def restore_memory_snapshot(self, snapshot, bSkipMappedFiles=True, bSkipOnError=False):
    """
        Attempts to restore the memory state as it was when the given snapshot
        was taken.

        @warning: Currently only the memory contents, state and protect bits
            are restored. Under some circumstances this method may fail (for
            example if memory was freed and then reused by a mapped file).

        @type  snapshot: list( L{win32.MemoryBasicInformation} )
        @param snapshot: Memory snapshot returned by L{take_memory_snapshot}.
            Snapshots returned by L{generate_memory_snapshot} don't work here.

        @type  bSkipMappedFiles: bool
        @param bSkipMappedFiles: C{True} to avoid restoring the contents of
            memory mapped files, C{False} otherwise. Use with care! Setting
            this to C{False} can cause undesired side effects - changes to
            memory mapped files may be written to disk by the OS. Also note
            that most mapped files are typically executables and don't change,
            so trying to restore their contents is usually a waste of time.

        @type  bSkipOnError: bool
        @param bSkipOnError: C{True} to issue a warning when an error occurs
            during the restoration of the snapshot, C{False} to stop and raise
            an exception instead. Use with care! Setting this to C{True} will
            cause the debugger to falsely believe the memory snapshot has been
            correctly restored.

        @raise WindowsError: An error occured while restoring the snapshot.
        @raise RuntimeError: An error occured while restoring the snapshot.
        @raise TypeError: A snapshot of the wrong type was passed.
        """
    if not snapshot or not isinstance(snapshot, list) or (not isinstance(snapshot[0], win32.MemoryBasicInformation)):
        raise TypeError('Only snapshots returned by take_memory_snapshot() can be used here.')
    hProcess = self.get_handle(win32.PROCESS_VM_WRITE | win32.PROCESS_VM_OPERATION | win32.PROCESS_SUSPEND_RESUME | win32.PROCESS_QUERY_INFORMATION)
    self.suspend()
    try:
        for old_mbi in snapshot:
            new_mbi = self.mquery(old_mbi.BaseAddress)
            if new_mbi.BaseAddress == old_mbi.BaseAddress and new_mbi.RegionSize == old_mbi.RegionSize:
                self.__restore_mbi(hProcess, new_mbi, old_mbi, bSkipMappedFiles)
            else:
                old_mbi = win32.MemoryBasicInformation(old_mbi)
                old_start = old_mbi.BaseAddress
                old_end = old_start + old_mbi.RegionSize
                new_start = new_mbi.BaseAddress
                new_end = new_start + new_mbi.RegionSize
                if old_start > new_start:
                    start = old_start
                else:
                    start = new_start
                if old_end < new_end:
                    end = old_end
                else:
                    end = new_end
                step = MemoryAddresses.pageSize
                old_mbi.RegionSize = step
                new_mbi.RegionSize = step
                address = start
                while address < end:
                    old_mbi.BaseAddress = address
                    new_mbi.BaseAddress = address
                    self.__restore_mbi(hProcess, new_mbi, old_mbi, bSkipMappedFiles, bSkipOnError)
                    address = address + step
    finally:
        self.resume()