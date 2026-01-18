import serial
from serial import (
from twisted.python.runtime import platform
def setRTS(self, on=1):
    self._serial.setRTS(on)