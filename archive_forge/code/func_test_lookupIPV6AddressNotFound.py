from __future__ import annotations
from typing_extensions import Protocol
from twisted.internet.defer import gatherResults
from twisted.names.dns import (
from twisted.names.hosts import Resolver, searchFileFor, searchFileForAll
from twisted.python.filepath import FilePath
from twisted.trial.unittest import SynchronousTestCase
def test_lookupIPV6AddressNotFound(self) -> None:
    """
        Like L{test_lookupAddressNotFound}, but for
        L{hosts.Resolver.lookupIPV6Address}.
        """
    self.failureResultOf(self.resolver.lookupIPV6Address(b'foueoa'), DomainError)