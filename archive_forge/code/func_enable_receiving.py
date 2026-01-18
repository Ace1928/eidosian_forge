import errno
import os
from functools import partial
from threading import Thread
from pyudev._os import pipe, poll
from pyudev._util import eintr_retry_call, ensure_byte_string
from pyudev.device import Device
def enable_receiving(self):
    """
        Switch the monitor into listing mode.

        Connect to the event source and receive incoming events.  Only after
        calling this method, the monitor listens for incoming events.

        .. note::

           This method is implicitly called by :meth:`__iter__`.  You don't
           need to call it explicitly, if you are iterating over the
           monitor.

        .. deprecated:: 0.16
           Will be removed in 1.0. Use :meth:`start()` instead.
        """
    import warnings
    warnings.warn('Will be removed in 1.0. Use Monitor.start() instead.', DeprecationWarning)
    self.start()