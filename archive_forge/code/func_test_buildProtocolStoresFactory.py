from twisted.internet import protocol
from twisted.python import failure
from twisted.trial import unittest
from twisted.words.xish import domish, utility, xmlstream
def test_buildProtocolStoresFactory(self):
    """
        The protocol factory is saved in the protocol.
        """
    xs = self.factory.buildProtocol(None)
    self.assertIdentical(self.factory, xs.factory)