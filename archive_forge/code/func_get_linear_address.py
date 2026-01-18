from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def get_linear_address(self, segment, address):
    """
        Translates segment-relative addresses to linear addresses.

        Linear addresses can be used to access a process memory,
        calling L{Process.read} and L{Process.write}.

        @type  segment: str
        @param segment: Segment register name.

        @type  address: int
        @param address: Segment relative memory address.

        @rtype:  int
        @return: Linear memory address.

        @raise ValueError: Address is too large for selector.

        @raise WindowsError:
            The current architecture does not support selectors.
            Selectors only exist in x86-based systems.
        """
    hThread = self.get_handle(win32.THREAD_QUERY_INFORMATION)
    selector = self.get_register(segment)
    ldt = win32.GetThreadSelectorEntry(hThread, selector)
    BaseLow = ldt.BaseLow
    BaseMid = ldt.HighWord.Bytes.BaseMid << 16
    BaseHi = ldt.HighWord.Bytes.BaseHi << 24
    Base = BaseLow | BaseMid | BaseHi
    LimitLow = ldt.LimitLow
    LimitHi = ldt.HighWord.Bits.LimitHi << 16
    Limit = LimitLow | LimitHi
    if address > Limit:
        msg = 'Address %s too large for segment %s (selector %d)'
        msg = msg % (HexDump.address(address, self.get_bits()), segment, selector)
        raise ValueError(msg)
    return Base + address