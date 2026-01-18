from __future__ import absolute_import
import errno
import logging
import select
import socket
import time
from serial.serialutil import SerialBase, SerialException, to_bytes, \
@property
def ri(self):
    """Read terminal status line: Ring Indicator"""
    if not self.is_open:
        raise PortNotOpenError()
    if self.logger:
        self.logger.info('returning dummy for ri')
    return False