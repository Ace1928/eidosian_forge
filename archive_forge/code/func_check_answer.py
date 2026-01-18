from __future__ import absolute_import
import logging
import socket
import struct
import threading
import time
import serial
from serial.serialutil import SerialBase, SerialException, to_bytes, \
def check_answer(self, suboption):
    """        Check an incoming subnegotiation block. The parameter already has
        cut off the header like sub option number and com port option value.
        """
    if self.value == suboption[:len(self.value)]:
        self.state = ACTIVE
    else:
        self.state = REALLY_INACTIVE
    if self.connection.logger:
        self.connection.logger.debug('SB Answer {} -> {!r} -> {}'.format(self.name, suboption, self.state))