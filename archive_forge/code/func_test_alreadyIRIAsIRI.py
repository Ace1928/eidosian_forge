from __future__ import annotations
from typing import Iterable
from typing_extensions import Protocol
from twisted.trial.unittest import SynchronousTestCase
from ..url import URL
def test_alreadyIRIAsIRI(self) -> None:
    """
        A L{URL} composed of non-ASCII text will result in non-ASCII text.
        """
    unicodey = 'http://é.com/é?á=í#ú'
    iri = URL.fromText(unicodey)
    alsoIRI = iri.asIRI()
    self.assertEqual(alsoIRI.asText(), unicodey)