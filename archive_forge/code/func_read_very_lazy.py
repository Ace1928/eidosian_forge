import sys
import socket
import selectors
from time import monotonic as _time
import warnings
def read_very_lazy(self):
    """Return any data available in the cooked queue (very lazy).

        Raise EOFError if connection closed and no data available.
        Return b'' if no cooked data available otherwise.  Don't block.

        """
    buf = self.cookedq
    self.cookedq = b''
    if not buf and self.eof and (not self.rawq):
        raise EOFError('telnet connection closed')
    return buf