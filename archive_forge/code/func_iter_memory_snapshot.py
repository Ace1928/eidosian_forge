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
def iter_memory_snapshot(self, minAddr=None, maxAddr=None):
    """
        Returns an iterator that allows you to go through the memory contents
        of a process.

        It's basically the same as the L{take_memory_snapshot} method, but it
        takes the snapshot of each memory region as it goes, as opposed to
        taking the whole snapshot at once. This allows you to work with very
        large snapshots without a significant performance penalty.

        Example::
            # Print the memory contents of a process.
            process.suspend()
            try:
                snapshot = process.generate_memory_snapshot()
                for mbi in snapshot:
                    print HexDump.hexblock(mbi.content, mbi.BaseAddress)
            finally:
                process.resume()

        The downside of this is the process must remain suspended while
        iterating the snapshot, otherwise strange things may happen.

        The snapshot can only iterated once. To be able to iterate indefinitely
        call the L{generate_memory_snapshot} method instead.

        You can also iterate the memory of a dead process, just as long as the
        last open handle to it hasn't been closed.

        @see: L{take_memory_snapshot}

        @type  minAddr: int
        @param minAddr: (Optional) Starting address in address range to query.

        @type  maxAddr: int
        @param maxAddr: (Optional) Ending address in address range to query.

        @rtype:  iterator of L{win32.MemoryBasicInformation}
        @return: Iterator of memory region information objects.
            Two extra properties are added to these objects:
             - C{filename}: Mapped filename, or C{None}.
             - C{content}: Memory contents, or C{None}.
        """
    memory = self.get_memory_map(minAddr, maxAddr)
    if not memory:
        return
    try:
        filenames = self.get_mapped_filenames(memory)
    except WindowsError:
        e = sys.exc_info()[1]
        if e.winerror != win32.ERROR_ACCESS_DENIED:
            raise
        filenames = dict()
    if minAddr is not None:
        minAddr = MemoryAddresses.align_address_to_page_start(minAddr)
        mbi = memory[0]
        if mbi.BaseAddress < minAddr:
            mbi.RegionSize = mbi.BaseAddress + mbi.RegionSize - minAddr
            mbi.BaseAddress = minAddr
    if maxAddr is not None:
        if maxAddr != MemoryAddresses.align_address_to_page_start(maxAddr):
            maxAddr = MemoryAddresses.align_address_to_page_end(maxAddr)
        mbi = memory[-1]
        if mbi.BaseAddress + mbi.RegionSize > maxAddr:
            mbi.RegionSize = maxAddr - mbi.BaseAddress
    while memory:
        mbi = memory.pop(0)
        mbi.filename = filenames.get(mbi.BaseAddress, None)
        if mbi.has_content():
            mbi.content = self.read(mbi.BaseAddress, mbi.RegionSize)
        else:
            mbi.content = None
        yield mbi