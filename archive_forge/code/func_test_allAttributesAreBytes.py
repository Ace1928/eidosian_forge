from twisted.python import urlpath
from twisted.trial import unittest
def test_allAttributesAreBytes(self):
    """
        A created L{URLPath} has bytes attributes.
        """
    self.assertIsInstance(self.path.scheme, bytes)
    self.assertIsInstance(self.path.netloc, bytes)
    self.assertIsInstance(self.path.path, bytes)
    self.assertIsInstance(self.path.query, bytes)
    self.assertIsInstance(self.path.fragment, bytes)