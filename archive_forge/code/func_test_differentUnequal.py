from __future__ import annotations
from typing import Iterable
from typing_extensions import Protocol
from twisted.trial.unittest import SynchronousTestCase
from ..url import URL
def test_differentUnequal(self) -> None:
    """
        Structurally different L{URL}s are unequal (C{!=}) to each other.
        """
    u1 = URL.fromText('http://localhost/a')
    u2 = URL.fromText('http://localhost/b')
    self.assertTrue(u1 != u2, f'{u1!r} == {u2!r}')