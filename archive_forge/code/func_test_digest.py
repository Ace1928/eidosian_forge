from zope.interface import implementer
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words.protocols.jabber import jid, sasl, sasl_mechanisms, xmlstream
from twisted.words.xish import domish
def test_digest(self):
    """
        Test setting DIGEST-MD5 as the authentication mechanism.
        """
    self.authenticator.jid = jid.JID('test@example.com')
    self.authenticator.password = 'secret'
    name = 'DIGEST-MD5'
    self.assertEqual(name, self._setMechanism(name))