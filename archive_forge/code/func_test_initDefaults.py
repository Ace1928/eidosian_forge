from __future__ import annotations
from typing import Iterable
from typing_extensions import Protocol
from twisted.trial.unittest import SynchronousTestCase
from ..url import URL
def test_initDefaults(self) -> None:
    """
        L{URL} should have appropriate default values.
        """

    def check(u: URL) -> None:
        self.assertUnicoded(u)
        self.assertURL(u, 'http', '', [], [], '', 80, '')
    check(URL('http', ''))
    check(URL('http', '', [], []))
    check(URL('http', '', [], [], ''))