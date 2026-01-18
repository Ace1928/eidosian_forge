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
@skipIf(*skipWhenNoSSL)
def test_certificateVerify(self):
    """
        The server certificate will be verified.
        """

    def fakeStartTLS(contextFactory):
        self.assertIsInstance(contextFactory, ClientTLSOptions)
        self.assertEqual(contextFactory._hostname, 'example.com')
        self.done.append('TLS')
    self.xmlstream.transport = proto_helpers.StringTransport()
    self.xmlstream.transport.startTLS = fakeStartTLS
    self.xmlstream.reset = lambda: self.done.append('reset')
    self.xmlstream.sendHeader = lambda: self.done.append('header')
    d = self.init.start()
    self.xmlstream.dataReceived("<proceed xmlns='%s'/>" % NS_XMPP_TLS)
    self.assertEqual(['TLS', 'reset', 'header'], self.done)
    return d