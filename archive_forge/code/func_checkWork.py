from zope.interface import implementer
from twisted.internet.interfaces import IConsumer, IPushProducer
import pywintypes
import win32api
import win32file
import win32pipe
def checkWork(self):
    numBytesWritten = 0
    if not self.outQueue:
        if self.disconnecting:
            self.writeConnectionLost()
            return 0
        try:
            win32file.WriteFile(self.writePipe, b'', None)
        except pywintypes.error:
            self.writeConnectionLost()
            return numBytesWritten
    while self.outQueue:
        data = self.outQueue.pop(0)
        errCode = 0
        try:
            errCode, nBytesWritten = win32file.WriteFile(self.writePipe, data, None)
        except win32api.error:
            self.writeConnectionLost()
            break
        else:
            numBytesWritten += nBytesWritten
            if len(data) > nBytesWritten:
                self.outQueue.insert(0, data[nBytesWritten:])
                break
    else:
        resumed = self.bufferEmpty()
        if not resumed and self.disconnecting:
            self.writeConnectionLost()
    return numBytesWritten