import copy
import struct
from io import BytesIO
from twisted.internet import protocol
from twisted.persisted import styles
from twisted.python import log
from twisted.python.compat import iterbytes
from twisted.python.reflect import fullyQualifiedName
def callExpressionReceived(self, obj):
    if self.currentDialect:
        self.expressionReceived(obj)
    elif self.isClient:
        for serverVer in obj:
            if serverVer in self.knownDialects:
                self.sendEncoded(serverVer)
                self._selectDialect(serverVer)
                break
        else:
            log.msg("The client doesn't speak any of the protocols offered by the server: disconnecting.")
            self.transport.loseConnection()
    elif obj in self.knownDialects:
        self._selectDialect(obj)
    else:
        log.msg("The client selected a protocol the server didn't suggest and doesn't know: disconnecting.")
        self.transport.loseConnection()