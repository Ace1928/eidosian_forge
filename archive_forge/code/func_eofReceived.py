from zope.interface import implementer
from twisted.internet import interfaces
from twisted.logger import Logger
from twisted.python import log
def eofReceived(self):
    """
        Called when the other side will send no more data.
        """
    self._log.info('remote eof')