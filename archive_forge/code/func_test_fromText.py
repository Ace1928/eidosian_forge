from __future__ import annotations
from typing import Iterable
from typing_extensions import Protocol
from twisted.trial.unittest import SynchronousTestCase
from ..url import URL
def test_fromText(self) -> None:
    """
        Round-tripping L{URL.fromText} with C{str} results in an equivalent
        URL.
        """
    urlpath = URL.fromText(theurl)
    self.assertEqual(theurl, urlpath.asText())