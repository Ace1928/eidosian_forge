from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def hook_function(self, pid, address, preCB=None, postCB=None, paramCount=None, signature=None):
    """
        Sets a function hook at the given address.

        If instead of an address you pass a label, the hook may be
        deferred until the DLL it points to is loaded.

        @type  pid: int
        @param pid: Process global ID.

        @type  address: int or str
        @param address:
            Memory address of code instruction to break at. It can be an
            integer value for the actual address or a string with a label
            to be resolved.

        @type  preCB: function
        @param preCB: (Optional) Callback triggered on function entry.

            The signature for the callback should be something like this::

                def pre_LoadLibraryEx(event, ra, lpFilename, hFile, dwFlags):

                    # return address
                    ra = params[0]

                    # function arguments start from here...
                    szFilename = event.get_process().peek_string(lpFilename)

                    # (...)

            Note that all pointer types are treated like void pointers, so your
            callback won't get the string or structure pointed to by it, but
            the remote memory address instead. This is so to prevent the ctypes
            library from being "too helpful" and trying to dereference the
            pointer. To get the actual data being pointed to, use one of the
            L{Process.read} methods.

        @type  postCB: function
        @param postCB: (Optional) Callback triggered on function exit.

            The signature for the callback should be something like this::

                def post_LoadLibraryEx(event, return_value):

                    # (...)

        @type  paramCount: int
        @param paramCount:
            (Optional) Number of parameters for the C{preCB} callback,
            not counting the return address. Parameters are read from
            the stack and assumed to be DWORDs in 32 bits and QWORDs in 64.

            This is a faster way to pull stack parameters in 32 bits, but in 64
            bits (or with some odd APIs in 32 bits) it won't be useful, since
            not all arguments to the hooked function will be of the same size.

            For a more reliable and cross-platform way of hooking use the
            C{signature} argument instead.

        @type  signature: tuple
        @param signature:
            (Optional) Tuple of C{ctypes} data types that constitute the
            hooked function signature. When the function is called, this will
            be used to parse the arguments from the stack. Overrides the
            C{paramCount} argument.

        @rtype:  bool
        @return: C{True} if the hook was set immediately, or C{False} if
            it was deferred.
        """
    try:
        aProcess = self.system.get_process(pid)
    except KeyError:
        aProcess = Process(pid)
    arch = aProcess.get_arch()
    hookObj = Hook(preCB, postCB, paramCount, signature, arch)
    bp = self.break_at(pid, address, hookObj)
    return bp is not None