from __future__ import annotations
from typing import Iterable
from typing_extensions import Protocol
from twisted.trial.unittest import SynchronousTestCase
from ..url import URL
def test_cloneUnchanged(self) -> None:
    """
        Verify that L{URL.replace} doesn't change any of the arguments it
        is passed.
        """
    urlpath = URL.fromText('https://x:1/y?z=1#A')
    self.assertEqual(urlpath.replace(urlpath.scheme, urlpath.host, urlpath.path, urlpath.query, urlpath.fragment, urlpath.port), urlpath)
    self.assertEqual(urlpath.replace(), urlpath)