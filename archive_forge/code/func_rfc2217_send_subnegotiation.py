from __future__ import absolute_import
import logging
import socket
import struct
import threading
import time
import serial
from serial.serialutil import SerialBase, SerialException, to_bytes, \
def rfc2217_send_subnegotiation(self, option, value=b''):
    """Subnegotiation of RFC 2217 parameters."""
    value = value.replace(IAC, IAC_DOUBLED)
    self.connection.write(IAC + SB + COM_PORT_OPTION + option + value + IAC + SE)