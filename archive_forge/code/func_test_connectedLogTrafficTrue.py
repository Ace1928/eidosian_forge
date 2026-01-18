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
def test_connectedLogTrafficTrue(self):
    """
        Test raw data functions set when logTraffic is set to True.
        """
    sm = self.streamManager
    sm.logTraffic = True
    handler = DummyXMPPHandler()
    handler.setHandlerParent(sm)
    xs = xmlstream.XmlStream(xmlstream.Authenticator())
    sm._connected(xs)
    self.assertNotIdentical(None, xs.rawDataInFn)
    self.assertNotIdentical(None, xs.rawDataOutFn)