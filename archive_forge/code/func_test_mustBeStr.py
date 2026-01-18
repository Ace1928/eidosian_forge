from twisted.python import urlpath
from twisted.trial import unittest
def test_mustBeStr(self):
    """
        C{URLPath.fromString} must take a L{str} or L{str} argument.
        """
    with self.assertRaises(ValueError):
        urlpath.URLPath.fromString(None)
    with self.assertRaises(ValueError):
        urlpath.URLPath.fromString(b'someurl')