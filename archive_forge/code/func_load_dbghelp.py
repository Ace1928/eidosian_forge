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
def load_dbghelp(cls, pathname=None):
    """
        Load the specified version of the C{dbghelp.dll} library.

        This library is shipped with the Debugging Tools for Windows, and it's
        required to load debug symbols.

        Normally you don't need to call this method, as WinAppDbg already tries
        to load the latest version automatically - but it may come in handy if
        the Debugging Tools are installed in a non standard folder.

        Example::
            from winappdbg import Debug

            def simple_debugger( argv ):

                # Instance a Debug object, passing it the event handler callback
                debug = Debug( my_event_handler )
                try:

                    # Load a specific dbghelp.dll file
                    debug.system.load_dbghelp("C:\\Some folder\\dbghelp.dll")

                    # Start a new process for debugging
                    debug.execv( argv )

                    # Wait for the debugee to finish
                    debug.loop()

                # Stop the debugger
                finally:
                    debug.stop()

        @see: U{http://msdn.microsoft.com/en-us/library/ms679294(VS.85).aspx}

        @type  pathname: str
        @param pathname:
            (Optional) Full pathname to the C{dbghelp.dll} library.
            If not provided this method will try to autodetect it.

        @rtype:  ctypes.WinDLL
        @return: Loaded instance of C{dbghelp.dll}.

        @raise NotImplementedError: This feature was not implemented for the
            current architecture.

        @raise WindowsError: An error occured while processing this request.
        """
    if not pathname:
        arch = win32.arch
        if arch == win32.ARCH_AMD64 and win32.bits == 32:
            arch = win32.ARCH_I386
        if not arch in cls.__dbghelp_locations:
            msg = 'Architecture %s is not currently supported.'
            raise NotImplementedError(msg % arch)
        found = []
        for pathname in cls.__dbghelp_locations[arch]:
            if path.isfile(pathname):
                try:
                    f_ver, p_ver = cls.get_file_version_info(pathname)[:2]
                except WindowsError:
                    msg = 'Failed to parse file version metadata for: %s'
                    warnings.warn(msg % pathname)
                if not f_ver:
                    f_ver = p_ver
                elif p_ver and p_ver > f_ver:
                    f_ver = p_ver
                found.append((f_ver, pathname))
        if found:
            found.sort()
            pathname = found.pop()[1]
        else:
            pathname = 'dbghelp.dll'
    dbghelp = ctypes.windll.LoadLibrary(pathname)
    ctypes.windll.dbghelp = dbghelp
    return dbghelp