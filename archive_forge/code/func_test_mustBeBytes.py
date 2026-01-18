from twisted.python import urlpath
from twisted.trial import unittest
def test_mustBeBytes(self):
    """
        L{URLPath.fromBytes} must take a L{bytes} argument.
        """
    with self.assertRaises(ValueError):
        urlpath.URLPath.fromBytes(None)
    with self.assertRaises(ValueError):
        urlpath.URLPath.fromBytes('someurl')