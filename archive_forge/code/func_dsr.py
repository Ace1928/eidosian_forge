from __future__ import absolute_import
import errno
import logging
import select
import socket
import time
from serial.serialutil import SerialBase, SerialException, to_bytes, \
@property
def dsr(self):
    """Read terminal status line: Data Set Ready"""
    if not self.is_open:
        raise PortNotOpenError()
    if self.logger:
        self.logger.info('returning dummy for dsr')
    return True