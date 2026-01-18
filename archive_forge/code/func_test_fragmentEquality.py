from __future__ import annotations
from typing import Iterable
from typing_extensions import Protocol
from twisted.trial.unittest import SynchronousTestCase
from ..url import URL
def test_fragmentEquality(self) -> None:
    """
        An URL created with the empty string for a fragment compares equal
        to an URL created with an unspecified fragment.
        """
    self.assertEqual(URL(fragment=''), URL())
    self.assertEqual(URL.fromText('http://localhost/#'), URL.fromText('http://localhost/'))