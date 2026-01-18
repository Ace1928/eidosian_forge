import sys
import socket
import selectors
from time import monotonic as _time
import warnings
def read_lazy(self):
    """Process and return data that's already in the queues (lazy).

        Raise EOFError if connection closed and no data available.
        Return b'' if no cooked data available otherwise.  Don't block
        unless in the midst of an IAC sequence.

        """
    self.process_rawq()
    return self.read_very_lazy()