from __future__ import annotations
from typing_extensions import Protocol
from twisted.internet.defer import gatherResults
from twisted.names.dns import (
from twisted.names.hosts import Resolver, searchFileFor, searchFileForAll
from twisted.python.filepath import FilePath
from twisted.trial.unittest import SynchronousTestCase
def test_caseInsensitively(self) -> None:
    """
        L{searchFileForAll} searches for names case-insensitively.
        """
    hosts = self.path()
    hosts.setContent(b'127.0.0.1     foobar.EXAMPLE.com\n')
    self.assertEqual(['127.0.0.1'], searchFileForAll(hosts, b'FOOBAR.example.com'))