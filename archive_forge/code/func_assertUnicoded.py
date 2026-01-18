from __future__ import annotations
from typing import Iterable
from typing_extensions import Protocol
from twisted.trial.unittest import SynchronousTestCase
from ..url import URL
def assertUnicoded(self, u: URL) -> None:
    """
        The given L{URL}'s components should be L{unicode}.

        @param u: The L{URL} to test.
        """
    self.assertIsInstance(u.scheme, str, repr(u))
    self.assertIsInstance(u.host, str, repr(u))
    for seg in u.path:
        self.assertIsInstance(seg, str, repr(u))
    for k, v in u.query:
        self.assertIsInstance(k, str, repr(u))
        self.assertTrue(v is None or isinstance(v, str), repr(u))
    self.assertIsInstance(u.fragment, str, repr(u))