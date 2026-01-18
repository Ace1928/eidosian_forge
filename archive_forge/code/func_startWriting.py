from zope.interface import implementer
from twisted.internet import interfaces
from twisted.logger import Logger
from twisted.python import log
def startWriting(self):
    """
        Called when the remote buffer has more room, as a hint to continue
        writing.
        """