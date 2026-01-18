from winappdbg import win32
from winappdbg import compat
from winappdbg.system import System
from winappdbg.textio import HexDump, CrashDump
from winappdbg.util import StaticClass, MemoryAddresses, PathOperations
import sys
import os
import time
import zlib
import warnings
def fetch_extra_data(self, event, takeMemorySnapshot=0):
    """
        Fetch extra data from the L{Event} object.

        @note: Since this method may take a little longer to run, it's best to
            call it only after you've determined the crash is interesting and
            you want to save it.

        @type  event: L{Event}
        @param event: Event object for crash.

        @type  takeMemorySnapshot: int
        @param takeMemorySnapshot:
            Memory snapshot behavior:
             - C{0} to take no memory information (default).
             - C{1} to take only the memory map.
               See L{Process.get_memory_map}.
             - C{2} to take a full memory snapshot.
               See L{Process.take_memory_snapshot}.
             - C{3} to take a live memory snapshot.
               See L{Process.generate_memory_snapshot}.
        """
    process = event.get_process()
    thread = event.get_thread()
    try:
        self.commandLine = process.get_command_line()
    except Exception:
        e = sys.exc_info()[1]
        warnings.warn('Cannot get command line, reason: %s' % str(e), CrashWarning)
    try:
        self.environmentData = process.get_environment_data()
        self.environment = process.parse_environment_data(self.environmentData)
    except Exception:
        e = sys.exc_info()[1]
        warnings.warn('Cannot get environment, reason: %s' % str(e), CrashWarning)
    self.registersPeek = thread.peek_pointers_in_registers()
    aModule = process.get_module_at_address(self.pc)
    if aModule is not None:
        self.modFileName = aModule.get_filename()
        self.lpBaseOfDll = aModule.get_base()
    try:
        self.stackRange = thread.get_stack_range()
    except Exception:
        e = sys.exc_info()[1]
        warnings.warn('Cannot get stack range, reason: %s' % str(e), CrashWarning)
    try:
        self.stackFrame = thread.get_stack_frame()
        stackFrame = self.stackFrame
    except Exception:
        self.stackFrame = thread.peek_stack_data()
        stackFrame = self.stackFrame[:64]
    if stackFrame:
        self.stackPeek = process.peek_pointers_in_data(stackFrame)
    self.faultCode = thread.peek_code_bytes()
    try:
        self.faultDisasm = thread.disassemble_around_pc(32)
    except Exception:
        e = sys.exc_info()[1]
        warnings.warn('Cannot disassemble, reason: %s' % str(e), CrashWarning)
    if self.eventCode == win32.EXCEPTION_DEBUG_EVENT:
        if self.pc != self.exceptionAddress and self.exceptionCode in (win32.EXCEPTION_ACCESS_VIOLATION, win32.EXCEPTION_ARRAY_BOUNDS_EXCEEDED, win32.EXCEPTION_DATATYPE_MISALIGNMENT, win32.EXCEPTION_IN_PAGE_ERROR, win32.EXCEPTION_STACK_OVERFLOW, win32.EXCEPTION_GUARD_PAGE):
            self.faultMem = process.peek(self.exceptionAddress, 64)
            if self.faultMem:
                self.faultPeek = process.peek_pointers_in_data(self.faultMem)
    if takeMemorySnapshot == 1:
        self.memoryMap = process.get_memory_map()
        mappedFilenames = process.get_mapped_filenames(self.memoryMap)
        for mbi in self.memoryMap:
            mbi.filename = mappedFilenames.get(mbi.BaseAddress, None)
            mbi.content = None
    elif takeMemorySnapshot == 2:
        self.memoryMap = process.take_memory_snapshot()
    elif takeMemorySnapshot == 3:
        self.memoryMap = process.generate_memory_snapshot()