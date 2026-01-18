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
@staticmethod
def fix_symbol_store_path(symbol_store_path=None, remote=True, force=False):
    """
        Fix the symbol store path. Equivalent to the C{.symfix} command in
        Microsoft WinDbg.

        If the symbol store path environment variable hasn't been set, this
        method will provide a default one.

        @type  symbol_store_path: str or None
        @param symbol_store_path: (Optional) Symbol store path to set.

        @type  remote: bool
        @param remote: (Optional) Defines the symbol store path to set when the
            C{symbol_store_path} is C{None}.

            If C{True} the default symbol store path is set to the Microsoft
            symbol server. Debug symbols will be downloaded through HTTP.
            This gives the best results but is also quite slow.

            If C{False} the default symbol store path is set to the local
            cache only. This prevents debug symbols from being downloaded and
            is faster, but unless you've installed the debug symbols on this
            machine or downloaded them in a previous debugging session, some
            symbols may be missing.

            If the C{symbol_store_path} argument is not C{None}, this argument
            is ignored entirely.

        @type  force: bool
        @param force: (Optional) If C{True} the new symbol store path is set
            always. If C{False} the new symbol store path is only set if
            missing.

            This allows you to call this method preventively to ensure the
            symbol server is always set up correctly when running your script,
            but without messing up whatever configuration the user has.

            Example::
                from winappdbg import Debug, System

                def simple_debugger( argv ):

                    # Instance a Debug object
                    debug = Debug( MyEventHandler() )
                    try:

                        # Make sure the remote symbol store is set
                        System.fix_symbol_store_path(remote = True,
                                                      force = False)

                        # Start a new process for debugging
                        debug.execv( argv )

                        # Wait for the debugee to finish
                        debug.loop()

                    # Stop the debugger
                    finally:
                        debug.stop()

        @rtype:  str or None
        @return: The previously set symbol store path if any,
            otherwise returns C{None}.
        """
    try:
        if symbol_store_path is None:
            local_path = 'C:\\SYMBOLS'
            if not path.isdir(local_path):
                local_path = 'C:\\Windows\\Symbols'
                if not path.isdir(local_path):
                    local_path = path.abspath('.')
            if remote:
                symbol_store_path = 'cache*;SRV*' + local_path + '*http://msdl.microsoft.com/download/symbols'
            else:
                symbol_store_path = 'cache*;SRV*' + local_path
        previous = os.environ.get('_NT_SYMBOL_PATH', None)
        if not previous or force:
            os.environ['_NT_SYMBOL_PATH'] = symbol_store_path
        return previous
    except Exception:
        e = sys.exc_info()[1]
        warnings.warn('Cannot fix symbol path, reason: %s' % str(e), RuntimeWarning)