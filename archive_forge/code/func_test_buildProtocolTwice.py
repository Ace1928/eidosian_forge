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
def test_buildProtocolTwice(self):
    """
        Subsequent calls to buildProtocol should result in different instances
        of the protocol, as well as their authenticators.
        """
    xs1 = self.factory.buildProtocol(None)
    xs2 = self.factory.buildProtocol(None)
    self.assertNotIdentical(xs1, xs2)
    self.assertNotIdentical(xs1.authenticator, xs2.authenticator)