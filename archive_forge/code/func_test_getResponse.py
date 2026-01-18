from twisted.python.compat import networkString
from twisted.trial import unittest
from twisted.words.protocols.jabber import sasl_mechanisms
def test_getResponse(self) -> None:
    """
        The response to a Digest-MD5 challenge includes the parameters from the
        challenge.
        """
    challenge = b'realm="localhost",nonce="1234",qop="auth",charset=utf-8,algorithm=md5-sess'
    directives = self.mechanism._parse(self.mechanism.getResponse(challenge))
    del directives[b'cnonce'], directives[b'response']
    self.assertEqual({b'username': b'test', b'nonce': b'1234', b'nc': b'00000001', b'qop': [b'auth'], b'charset': b'utf-8', b'realm': b'localhost', b'digest-uri': b'xmpp/example.org'}, directives)