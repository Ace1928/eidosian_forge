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
def set_low_latency_mode(self, low_latency_settings):
    buf = array.array('i', [0] * 32)
    try:
        fcntl.ioctl(self.fd, termios.TIOCGSERIAL, buf)
        if low_latency_settings:
            buf[4] |= 8192
        else:
            buf[4] &= ~8192
        fcntl.ioctl(self.fd, termios.TIOCSSERIAL, buf)
    except IOError as e:
        raise ValueError('Failed to update ASYNC_LOW_LATENCY flag to {}: {}'.format(low_latency_settings, e))