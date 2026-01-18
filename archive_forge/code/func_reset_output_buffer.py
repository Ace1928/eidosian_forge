from __future__ import absolute_import
import errno
import logging
import select
import socket
import time
from serial.serialutil import SerialBase, SerialException, to_bytes, \
def reset_output_buffer(self):
    """        Clear output buffer, aborting the current output and
        discarding all that is in the buffer.
        """
    if not self.is_open:
        raise PortNotOpenError()
    if self.logger:
        self.logger.info('ignored reset_output_buffer')