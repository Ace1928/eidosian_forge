from __future__ import absolute_import
import logging
import socket
import struct
import threading
import time
import serial
from serial.serialutil import SerialBase, SerialException, to_bytes, \
def telnet_send_option(self, action, option):
    """Send DO, DONT, WILL, WONT."""
    self.connection.write(IAC + action + option)