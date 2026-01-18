import win32event
import win32file
from serial import EIGHTBITS, PARITY_NONE, STOPBITS_ONE
from serial.serialutil import to_bytes
from twisted.internet import abstract
from twisted.internet.serialport import BaseSerialPort
def serialWriteEvent(self):
    try:
        dataToWrite = self.outQueue.pop(0)
    except IndexError:
        self.writeInProgress = 0
        return
    else:
        win32file.WriteFile(self._serial._port_handle, dataToWrite, self._overlappedWrite)