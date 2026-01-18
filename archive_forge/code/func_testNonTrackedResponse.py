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
def testNonTrackedResponse(self):
    """
        Test that untracked iq responses don't trigger any action.

        Untracked means that the id of the incoming response iq is not
        in the stream's C{iqDeferreds} dictionary.
        """
    xs = self.xmlstream
    xmlstream.upgradeWithIQResponseTracker(xs)
    self.assertFalse(xs.iqDeferreds)

    def cb(iq):
        self.assertFalse(getattr(iq, 'handled', False))
    xs.addObserver('/iq', cb, -1)
    xs.dataReceived("<iq type='result' id='test'/>")