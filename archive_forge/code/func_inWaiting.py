import serial
from serial import (
from twisted.python.runtime import platform
def inWaiting(self):
    return self._serial.inWaiting()