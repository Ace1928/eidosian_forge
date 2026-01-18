from hashlib import sha1
from zope.interface.verify import verifyObject
from twisted.python import failure
from twisted.trial import unittest
from twisted.words.protocols.jabber import component, ijabber, xmlstream
from twisted.words.protocols.jabber.jid import JID
from twisted.words.xish import domish
from twisted.words.xish.utility import XmlPipe
def test_makeConnection(self):
    """
        A new connection increases the stream serial count. No logs by default.
        """
    self.xmlstream.dispatch(self.xmlstream, xmlstream.STREAM_CONNECTED_EVENT)
    self.assertEqual(0, self.xmlstream.serial)
    self.assertEqual(1, self.factory.serial)
    self.assertIdentical(None, self.xmlstream.rawDataInFn)
    self.assertIdentical(None, self.xmlstream.rawDataOutFn)