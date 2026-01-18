from urllib.parse import urlparse
from twisted.trial import unittest
from twisted.web import client
def test_noFragments(self):
    """
        L{client._urljoin} does not include a fragment identifier in the
        resulting URL if neither the base nor the new path include a fragment
        identifier.
        """
    self.assertEqual(client._urljoin(b'http://foo.com/bar', b'/quux'), b'http://foo.com/quux')
    self.assertEqual(client._urljoin(b'http://foo.com/bar#', b'/quux'), b'http://foo.com/quux')
    self.assertEqual(client._urljoin(b'http://foo.com/bar', b'/quux#'), b'http://foo.com/quux')