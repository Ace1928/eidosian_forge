from __future__ import annotations
from typing_extensions import Protocol
from twisted.internet.defer import gatherResults
from twisted.names.dns import (
from twisted.names.hosts import Resolver, searchFileFor, searchFileForAll
from twisted.python.filepath import FilePath
from twisted.trial.unittest import SynchronousTestCase
def test_findAddress(self) -> None:
    """
        If there is an IPv4 address for the hostname passed to L{searchFileFor},
        it is returned.
        """
    hosts = self.path()
    hosts.setContent(b'10.2.3.4 foo.example.com\n')
    self.assertEqual('10.2.3.4', searchFileFor(hosts.path, b'foo.example.com'))