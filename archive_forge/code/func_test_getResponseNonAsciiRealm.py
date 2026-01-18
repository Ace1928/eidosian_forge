from twisted.python.compat import networkString
from twisted.trial import unittest
from twisted.words.protocols.jabber import sasl_mechanisms
def test_getResponseNonAsciiRealm(self) -> None:
    """
        Bytes outside the ASCII range in the challenge are nevertheless
        included in the response.
        """
    challenge = b'realm="\xc3\xa9chec.example.org",nonce="1234",qop="auth",charset=utf-8,algorithm=md5-sess'
    directives = self.mechanism._parse(self.mechanism.getResponse(challenge))
    del directives[b'cnonce'], directives[b'response']
    self.assertEqual({b'username': b'test', b'nonce': b'1234', b'nc': b'00000001', b'qop': [b'auth'], b'charset': b'utf-8', b'realm': b'\xc3\xa9chec.example.org', b'digest-uri': b'xmpp/example.org'}, directives)