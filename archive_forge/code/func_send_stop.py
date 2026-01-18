import errno
import os
from functools import partial
from threading import Thread
from pyudev._os import pipe, poll
from pyudev._util import eintr_retry_call, ensure_byte_string
from pyudev.device import Device
def send_stop(self):
    """
        Send a stop signal to the background thread.

        The background thread will eventually exit, but it may still be running
        when this method returns.  This method is essentially the asynchronous
        equivalent to :meth:`stop()`.

        .. note::

           The underlying :attr:`monitor` is *not* stopped.
        """
    if self._stop_event is None:
        return
    with self._stop_event.sink:
        eintr_retry_call(self._stop_event.sink.write, b'\x01')
        self._stop_event.sink.flush()