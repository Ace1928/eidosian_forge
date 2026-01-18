from __future__ import absolute_import
import errno
import fcntl
import os
import select
import struct
import sys
import termios
import serial
from serial.serialutil import SerialBase, SerialException, to_bytes, \
def set_input_flow_control(self, enable=True):
    """        Manually control flow - when software flow control is enabled.
        This will send XON (true) or XOFF (false) to the other device.
        WARNING: this function is not portable to different platforms!
        """
    if not self.is_open:
        raise PortNotOpenError()
    if enable:
        termios.tcflow(self.fd, termios.TCION)
    else:
        termios.tcflow(self.fd, termios.TCIOFF)