from twisted.python import urlpath
from twisted.trial import unittest
def test_nonASCIICharacters(self):
    """
        L{URLPath.fromString} can load non-ASCII characters.
        """
    url = urlpath.URLPath.fromString('http://example.com/ÿ\x00')
    self.assertEqual(str(url), 'http://example.com/%C3%BF%00')