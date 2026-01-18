from __future__ import annotations
from typing_extensions import Protocol
from twisted.internet.defer import gatherResults
from twisted.names.dns import (
from twisted.names.hosts import Resolver, searchFileFor, searchFileForAll
from twisted.python.filepath import FilePath
from twisted.trial.unittest import SynchronousTestCase
def test_malformedIP(self) -> None:
    """
        L{searchFileForAll} ignores any malformed IP addresses associated with
        the name passed to it.
        """
    hosts = self.path()
    hosts.setContent(b'127.0.0.1\tmiser.example.org\tmiser\nnot-an-ip\tmiser\n\xffnot-ascii\t miser\n# miser\nmiser\n::1 miser')
    self.assertEqual(['127.0.0.1', '::1'], searchFileForAll(hosts, b'miser'))