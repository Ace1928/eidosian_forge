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
def inject_dll(self, dllname, procname=None, lpParameter=0, bWait=True, dwTimeout=None):
    """
        Injects a DLL into the process memory.

        @warning: Setting C{bWait} to C{True} when the process is frozen by a
            debug event will cause a deadlock in your debugger.

        @warning: This involves allocating memory in the target process.
            This is how the freeing of this memory is handled:

             - If the C{bWait} flag is set to C{True} the memory will be freed
               automatically before returning from this method.
             - If the C{bWait} flag is set to C{False}, the memory address is
               set as the L{Thread.pInjectedMemory} property of the returned
               thread object.
             - L{Debug} objects free L{Thread.pInjectedMemory} automatically
               both when it detaches from a process and when the injected
               thread finishes its execution.
             - The {Thread.kill} method also frees L{Thread.pInjectedMemory}
               automatically, even if you're not attached to the process.

            You could still be leaking memory if not careful. For example, if
            you inject a dll into a process you're not attached to, you don't
            wait for the thread's completion and you don't kill it either, the
            memory would be leaked.

        @see: L{inject_code}

        @type  dllname: str
        @param dllname: Name of the DLL module to load.

        @type  procname: str
        @param procname: (Optional) Procedure to call when the DLL is loaded.

        @type  lpParameter: int
        @param lpParameter: (Optional) Parameter to the C{procname} procedure.

        @type  bWait: bool
        @param bWait: C{True} to wait for the process to finish.
            C{False} to return immediately.

        @type  dwTimeout: int
        @param dwTimeout: (Optional) Timeout value in milliseconds.
            Ignored if C{bWait} is C{False}.

        @rtype: L{Thread}
        @return: Newly created thread object. If C{bWait} is set to C{True} the
            thread will be dead, otherwise it will be alive.

        @raise NotImplementedError: The target platform is not supported.
            Currently calling a procedure in the library is only supported in
            the I{i386} architecture.

        @raise WindowsError: An exception is raised on error.
        """
    aModule = self.get_module_by_name(compat.b('kernel32.dll'))
    if aModule is None:
        self.scan_modules()
        aModule = self.get_module_by_name(compat.b('kernel32.dll'))
    if aModule is None:
        raise RuntimeError('Cannot resolve kernel32.dll in the remote process')
    if procname:
        if self.get_arch() != win32.ARCH_I386:
            raise NotImplementedError()
        dllname = compat.b(dllname)
        pllib = aModule.resolve(compat.b('LoadLibraryA'))
        if not pllib:
            raise RuntimeError('Cannot resolve kernel32.dll!LoadLibraryA in the remote process')
        pgpad = aModule.resolve(compat.b('GetProcAddress'))
        if not pgpad:
            raise RuntimeError('Cannot resolve kernel32.dll!GetProcAddress in the remote process')
        pvf = aModule.resolve(compat.b('VirtualFree'))
        if not pvf:
            raise RuntimeError('Cannot resolve kernel32.dll!VirtualFree in the remote process')
        code = compat.b('')
        code += compat.b('è') + struct.pack('<L', len(dllname) + 1) + dllname + compat.b('\x00')
        code += compat.b('¸') + struct.pack('<L', pllib)
        code += compat.b('ÿÐ')
        if procname:
            code += compat.b('è') + struct.pack('<L', len(procname) + 1)
            code += procname + compat.b('\x00')
            code += compat.b('P')
            code += compat.b('¸') + struct.pack('<L', pgpad)
            code += compat.b('ÿÐ')
            code += compat.b('\x8bì')
            code += compat.b('h') + struct.pack('<L', lpParameter)
            code += compat.b('ÿÐ')
            code += compat.b('\x8bå')
        code += compat.b('Z')
        code += compat.b('h') + struct.pack('<L', win32.MEM_RELEASE)
        code += compat.b('h') + struct.pack('<L', 4096)
        code += compat.b('è\x00\x00\x00\x00')
        code += compat.b('\x81$$\x00ðÿÿ')
        code += compat.b('¸') + struct.pack('<L', pvf)
        code += compat.b('R')
        code += compat.b('ÿà')
        aThread, lpStartAddress = self.inject_code(code, lpParameter)
    else:
        if type(dllname) == type(u''):
            pllibname = compat.b('LoadLibraryW')
            bufferlen = (len(dllname) + 1) * 2
            dllname = win32.ctypes.create_unicode_buffer(dllname).raw[:bufferlen + 1]
        else:
            pllibname = compat.b('LoadLibraryA')
            dllname = compat.b(dllname) + compat.b('\x00')
            bufferlen = len(dllname)
        pllib = aModule.resolve(pllibname)
        if not pllib:
            msg = 'Cannot resolve kernel32.dll!%s in the remote process'
            raise RuntimeError(msg % pllibname)
        pbuffer = self.malloc(bufferlen)
        try:
            self.write(pbuffer, dllname)
            try:
                aThread = self.start_thread(pllib, pbuffer)
            except WindowsError:
                e = sys.exc_info()[1]
                if e.winerror != win32.ERROR_NOT_ENOUGH_MEMORY:
                    raise
                raise NotImplementedError('Target process belongs to a different Terminal Services session, cannot inject!')
            aThread.pInjectedMemory = pbuffer
        except Exception:
            self.free(pbuffer)
            raise
    if bWait:
        aThread.wait(dwTimeout)
        self.free(aThread.pInjectedMemory)
        del aThread.pInjectedMemory
    return aThread