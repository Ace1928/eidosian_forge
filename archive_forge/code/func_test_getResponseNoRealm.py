from twisted.python.compat import networkString
from twisted.trial import unittest
from twisted.words.protocols.jabber import sasl_mechanisms
def test_getResponseNoRealm(self) -> None:
    """
        The response to a challenge without a realm uses the host part of the
        JID as the realm.
        """
    challenge = b'nonce="1234",qop="auth",charset=utf-8,algorithm=md5-sess'
    directives = self.mechanism._parse(self.mechanism.getResponse(challenge))
    self.assertEqual(directives[b'realm'], b'example.org')