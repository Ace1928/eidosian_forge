from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def peek_pointers_in_registers(self, peekSize=16, context=None):
    """
        Tries to guess which values in the registers are valid pointers,
        and reads some data from them.

        @type  peekSize: int
        @param peekSize: Number of bytes to read from each pointer found.

        @type  context: dict( str S{->} int )
        @param context: (Optional)
            Dictionary mapping register names to their values.
            If not given, the current thread context will be used.

        @rtype:  dict( str S{->} str )
        @return: Dictionary mapping register names to the data they point to.
        """
    peekable_registers = ('Eax', 'Ebx', 'Ecx', 'Edx', 'Esi', 'Edi', 'Ebp')
    if not context:
        context = self.get_context(win32.CONTEXT_CONTROL | win32.CONTEXT_INTEGER)
    aProcess = self.get_process()
    data = dict()
    for reg_name, reg_value in compat.iteritems(context):
        if reg_name not in peekable_registers:
            continue
        reg_data = aProcess.peek(reg_value, peekSize)
        if reg_data:
            data[reg_name] = reg_data
    return data