from __future__ import absolute_import
import logging
import numbers
import time
from serial.serialutil import SerialBase, SerialException, to_bytes, iterbytes, SerialTimeoutException, PortNotOpenError
@property
def out_waiting(self):
    """Return how many bytes the in the outgoing buffer"""
    if not self.is_open:
        raise PortNotOpenError()
    if self.logger:
        self.logger.debug('out_waiting -> {:d}'.format(self.queue.qsize()))
    return self.queue.qsize()