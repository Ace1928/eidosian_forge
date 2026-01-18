from hashlib import sha1
from unittest import skipIf
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words.protocols.jabber import client, error, jid, xmlstream
from twisted.words.protocols.jabber.sasl import SASLInitiatingInitializer
from twisted.words.xish import utility
def testFailRequestFields(self):
    """
        Test initializer failure of request for fields for authentication.
        """

    def onAuthGet(iq):
        """
            Called when the initializer sent a query for authentication methods.

            The server responds that the client is not authorized to authenticate.
            """
        response = error.StanzaError('not-authorized').toResponse(iq)
        self.pipe.source.send(response)
    d1 = self.waitFor(IQ_AUTH_GET, onAuthGet)
    d2 = self.init.initialize()
    self.assertFailure(d2, error.StanzaError)
    return defer.gatherResults([d1, d2])