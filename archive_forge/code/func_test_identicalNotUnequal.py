from __future__ import annotations
from typing import Iterable
from typing_extensions import Protocol
from twisted.trial.unittest import SynchronousTestCase
from ..url import URL
def test_identicalNotUnequal(self) -> None:
    """
        Identical L{URL}s are not unequal (C{!=}) to each other.
        """
    u = URL.fromText('http://localhost/')
    self.assertFalse(u != u, '%r == itself' % u)