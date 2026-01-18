from __future__ import annotations
from typing_extensions import Protocol
from twisted.internet.defer import gatherResults
from twisted.names.dns import (
from twisted.names.hosts import Resolver, searchFileFor, searchFileForAll
from twisted.python.filepath import FilePath
from twisted.trial.unittest import SynchronousTestCase
def test_firstAddress(self) -> None:
    """
        The first address associated with the given hostname is returned.
        """
    hosts = self.path()
    hosts.setContent(b'::1 foo.example.com\n10.1.2.3 foo.example.com\nfe80::21b:fcff:feee:5a1d foo.example.com\n')
    self.assertEqual('::1', searchFileFor(hosts.path, b'foo.example.com'))