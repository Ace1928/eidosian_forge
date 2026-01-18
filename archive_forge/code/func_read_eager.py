import sys
import socket
import selectors
from time import monotonic as _time
import warnings
def read_eager(self):
    """Read readily available data.

        Raise EOFError if connection closed and no cooked data
        available.  Return b'' if no cooked data available otherwise.
        Don't block unless in the midst of an IAC sequence.

        """
    self.process_rawq()
    while not self.cookedq and (not self.eof) and self.sock_avail():
        self.fill_rawq()
        self.process_rawq()
    return self.read_very_lazy()