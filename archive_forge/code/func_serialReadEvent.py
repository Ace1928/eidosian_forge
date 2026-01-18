import win32event
import win32file
from serial import EIGHTBITS, PARITY_NONE, STOPBITS_ONE
from serial.serialutil import to_bytes
from twisted.internet import abstract
from twisted.internet.serialport import BaseSerialPort
def serialReadEvent(self):
    n = win32file.GetOverlappedResult(self._serial._port_handle, self._overlappedRead, 0)
    first = to_bytes(self.read_buf[:n])
    flags, comstat = self._clearCommError()
    if comstat.cbInQue:
        win32event.ResetEvent(self._overlappedRead.hEvent)
        rc, buf = win32file.ReadFile(self._serial._port_handle, win32file.AllocateReadBuffer(comstat.cbInQue), self._overlappedRead)
        n = win32file.GetOverlappedResult(self._serial._port_handle, self._overlappedRead, 1)
        self.protocol.dataReceived(first + to_bytes(buf[:n]))
    else:
        self.protocol.dataReceived(first)
    win32event.ResetEvent(self._overlappedRead.hEvent)
    rc, self.read_buf = win32file.ReadFile(self._serial._port_handle, win32file.AllocateReadBuffer(1), self._overlappedRead)