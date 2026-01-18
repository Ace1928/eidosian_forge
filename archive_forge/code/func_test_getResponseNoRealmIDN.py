from twisted.python.compat import networkString
from twisted.trial import unittest
from twisted.words.protocols.jabber import sasl_mechanisms
def test_getResponseNoRealmIDN(self) -> None:
    """
        If the challenge does not include a realm and the host part of the JID
        includes bytes outside of the ASCII range, the response still includes
        the host part of the JID as the realm.
        """
    self.mechanism = sasl_mechanisms.DigestMD5('xmpp', 'échec.example.org', None, 'test', 'secret')
    challenge = b'nonce="1234",qop="auth",charset=utf-8,algorithm=md5-sess'
    directives = self.mechanism._parse(self.mechanism.getResponse(challenge))
    self.assertEqual(directives[b'realm'], b'\xc3\xa9chec.example.org')