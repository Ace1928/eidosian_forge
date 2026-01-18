from __future__ import annotations
from typing import Iterable
from typing_extensions import Protocol
from twisted.trial.unittest import SynchronousTestCase
from ..url import URL
def test_identicalEqual(self) -> None:
    """
        L{URL} compares equal to itself.
        """
    u = URL.fromText('http://localhost/')
    self.assertEqual(u, u)