from hashlib import sha1
from zope.interface.verify import verifyObject
from twisted.python import failure
from twisted.trial import unittest
from twisted.words.protocols.jabber import component, ijabber, xmlstream
from twisted.words.protocols.jabber.jid import JID
from twisted.words.xish import domish
from twisted.words.xish.utility import XmlPipe
def testHandshake(self):
    """
        Test basic operations of component handshake.
        """
    d = self.init.initialize()
    handshake = self.output[-1]
    self.assertEqual('handshake', handshake.name)
    self.assertEqual('test:component', handshake.uri)
    self.assertEqual(sha1(b'12345' + b'secret').hexdigest(), str(handshake))
    handshake.children = []
    self.xmlstream.dataReceived(handshake.toXml())
    return d