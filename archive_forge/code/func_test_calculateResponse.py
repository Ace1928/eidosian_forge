from twisted.python.compat import networkString
from twisted.trial import unittest
from twisted.words.protocols.jabber import sasl_mechanisms
def test_calculateResponse(self) -> None:
    """
        The response to a Digest-MD5 challenge is computed according to RFC
        2831.
        """
    charset = 'utf-8'
    nonce = b'OA6MG9tEQGm2hh'
    nc = networkString(f'{1:08x}')
    cnonce = b'OA6MHXh6VqTrRk'
    username = 'Иchris'
    password = 'Иsecret'
    host = 'Иelwood.innosoft.com'
    digestURI = 'imap/Иelwood.innosoft.com'.encode(charset)
    mechanism = sasl_mechanisms.DigestMD5(b'imap', host, None, username, password)
    response = mechanism._calculateResponse(cnonce, nc, nonce, username.encode(charset), password.encode(charset), host.encode(charset), digestURI)
    self.assertEqual(response, b'7928f233258be88392424d094453c5e3')