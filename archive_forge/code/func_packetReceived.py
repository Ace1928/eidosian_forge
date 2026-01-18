from typing import Dict
from twisted.logger import Logger
def packetReceived(self, messageNum, packet):
    """
        called when we receive a packet on the transport
        """
    if messageNum in self.protocolMessages:
        messageType = self.protocolMessages[messageNum]
        f = getattr(self, 'ssh_%s' % messageType[4:], None)
        if f is not None:
            return f(packet)
    self._log.info("couldn't handle {messageNum} {packet!r}", messageNum=messageNum, packet=packet)
    self.transport.sendUnimplemented()