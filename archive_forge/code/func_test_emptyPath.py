from urllib.parse import urlparse
from twisted.trial import unittest
from twisted.web import client
def test_emptyPath(self):
    """
        The path of a I{URI} with an empty path is C{b'/'}.
        """
    uri = self.makeURIString(b'http://HOST/')
    self.assertURIEquals(client.URI.fromBytes(uri), scheme=b'http', netloc=self.uriHost, host=self.host, port=80, path=b'/')