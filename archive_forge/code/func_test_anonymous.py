from zope.interface import implementer
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words.protocols.jabber import jid, sasl, sasl_mechanisms, xmlstream
from twisted.words.xish import domish
def test_anonymous(self):
    """
        Test setting ANONYMOUS as the authentication mechanism.
        """
    self.authenticator.jid = jid.JID('example.com')
    self.authenticator.password = None
    name = 'ANONYMOUS'
    self.assertEqual(name, self._setMechanism(name))