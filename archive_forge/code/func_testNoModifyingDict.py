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
def testNoModifyingDict(self):
    """
        Test to make sure the errbacks cannot cause the iteration of the
        iqDeferreds to blow up in our face.
        """

    def eb(failure):
        d = xmlstream.IQ(self.xmlstream).send()
        d.addErrback(eb)
    d = self.iq.send()
    d.addErrback(eb)
    self.xmlstream.connectionLost('Closed by peer')
    return d