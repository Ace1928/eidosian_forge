from __future__ import annotations
from typing import Iterable
from typing_extensions import Protocol
from twisted.trial.unittest import SynchronousTestCase
from ..url import URL
def test_alreadyURIAsURI(self) -> None:
    """
        A L{URL} composed of encoded text will remain encoded.
        """
    expectedURI = 'http://xn--9ca.com/%C3%A9?%C3%A1=%C3%AD#%C3%BA'
    uri = URL.fromText(expectedURI)
    actualURI = uri.asURI().asText()
    self.assertEqual(actualURI, expectedURI)