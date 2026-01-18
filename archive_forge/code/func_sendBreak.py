import serial
from serial import (
from twisted.python.runtime import platform
def sendBreak(self):
    self._serial.sendBreak()