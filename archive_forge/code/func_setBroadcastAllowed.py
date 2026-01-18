from random import randrange
from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.internet.address import IPv4Address
from twisted.internet.defer import succeed
from twisted.internet.interfaces import IReactorUDP, IUDPTransport
from twisted.internet.task import Clock
def setBroadcastAllowed(self, enabled):
    """
        Dummy implementation to satisfy L{IUDPTransport}.
        """
    pass