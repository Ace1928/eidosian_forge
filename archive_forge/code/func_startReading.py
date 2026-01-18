import errno
from zope.interface import implementer
from twisted.internet import error, interfaces, main
from twisted.internet.abstract import _ConsumerMixin, _dataMustBeBytes, _LogOwner
from twisted.internet.iocpreactor import iocpsupport as _iocp
from twisted.internet.iocpreactor.const import ERROR_HANDLE_EOF, ERROR_IO_PENDING
from twisted.python import failure
def startReading(self):
    self.reactor.addActiveHandle(self)
    if not self._readScheduled and (not self.reading):
        self.reading = True
        self._readScheduled = self.reactor.callLater(0, self._resumeReading)