from __future__ import absolute_import
import errno
import logging
import select
import socket
import time
from serial.serialutil import SerialBase, SerialException, to_bytes, \
def send_break(self, duration=0.25):
    """        Send break condition. Timed, returns to idle state after given
        duration.
        """
    if not self.is_open:
        raise PortNotOpenError()
    if self.logger:
        self.logger.info('ignored send_break({!r})'.format(duration))