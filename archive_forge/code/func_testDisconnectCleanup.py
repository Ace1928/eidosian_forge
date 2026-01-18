from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.internet import defer, task
from twisted.internet.error import ConnectionLost
from twisted.internet.interfaces import IProtocolFactory
from twisted.python import failure
from twisted.test import proto_helpers
from twisted.trial import unittest
from twisted.words.protocols.jabber import error, ijabber, jid, xmlstream
from twisted.words.test.test_xmlstream import GenericXmlStreamFactoryTestsMixin
from twisted.words.xish import domish
def testDisconnectCleanup(self):
    """
        Test if deferreds for iq's that haven't yet received a response
        have their errback called on stream disconnect.
        """
    d = self.iq.send()
    xs = self.xmlstream
    xs.connectionLost('Closed by peer')
    self.assertFailure(d, ConnectionLost)
    return d