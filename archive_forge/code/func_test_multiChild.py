from __future__ import annotations
from typing import Iterable
from typing_extensions import Protocol
from twisted.trial.unittest import SynchronousTestCase
from ..url import URL
def test_multiChild(self) -> None:
    """
        L{URL.child} receives multiple segments as C{*args} and appends each in
        turn.
        """
    self.assertEqual(URL.fromText('http://example.com/a/b').child('c', 'd', 'e').asText(), 'http://example.com/a/b/c/d/e')