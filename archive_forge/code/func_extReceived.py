from zope.interface import implementer
from twisted.internet import interfaces
from twisted.logger import Logger
from twisted.python import log
def extReceived(self, dataType, data):
    """
        Called when we receive extended data (usually standard error).

        @type dataType: L{int}
        @type data:     L{str}
        """
    self._log.debug('got extended data {dataType} {data!r}', dataType=dataType, data=data)