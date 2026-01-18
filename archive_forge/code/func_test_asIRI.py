from __future__ import annotations
from typing import Iterable
from typing_extensions import Protocol
from twisted.trial.unittest import SynchronousTestCase
from ..url import URL
def test_asIRI(self) -> None:
    """
        L{URL.asIRI} decodes any percent-encoded text in the URI, making it
        more suitable for reading by humans, and returns a new L{URL}.
        """
    asciiish = 'http://xn--9ca.com/%C3%A9?%C3%A1=%C3%AD#%C3%BA'
    uri = URL.fromText(asciiish)
    iri = uri.asIRI()
    self.assertEqual(uri.host, 'xn--9ca.com')
    self.assertEqual(uri.path[0], '%C3%A9')
    self.assertEqual(uri.asText(), asciiish)
    expectedIRI = 'http://é.com/é?á=í#ú'
    actualIRI = iri.asText()
    self.assertEqual(actualIRI, expectedIRI, f'{actualIRI!r} != {expectedIRI!r}')