from __future__ import absolute_import
import sys
import time
import serial
from serial.serialutil import  to_bytes
def cancel_read(self):
    self.formatter.control('Q-RX', 'cancel_read')
    super(Serial, self).cancel_read()