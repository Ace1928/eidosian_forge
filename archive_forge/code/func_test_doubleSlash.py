from twisted.python import urlpath
from twisted.trial import unittest
def test_doubleSlash(self):
    """
        Calling L{urlpath.URLPath.click} on a L{urlpath.URLPath} with a
        trailing slash with a relative URL containing a leading slash will
        result in a URL with a single slash at the start of the path portion.
        """
    self.assertEqual(str(self.path.click(b'/hello/world')).encode('ascii'), b'http://example.com/hello/world')