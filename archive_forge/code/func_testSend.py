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
def testSend(self):
    self.xmlstream.transport.clear()
    self.iq.send()
    idBytes = self.iq['id'].encode('utf-8')
    self.assertIn(self.xmlstream.transport.value(), [b"<iq type='get' id='" + idBytes + b"'/>", b"<iq id='" + idBytes + b"' type='get'/>"])