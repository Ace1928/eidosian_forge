from __future__ import annotations
from typing_extensions import Protocol
from twisted.internet.defer import gatherResults
from twisted.names.dns import (
from twisted.names.hosts import Resolver, searchFileFor, searchFileForAll
from twisted.python.filepath import FilePath
from twisted.trial.unittest import SynchronousTestCase
def test_lookupAddressNotFound(self) -> None:
    """
        L{hosts.Resolver.lookupAddress} returns a L{Deferred} which fires with
        L{dns.DomainError} if the name passed in has no addresses in the hosts
        file.
        """
    self.failureResultOf(self.resolver.lookupAddress(b'foueoa'), DomainError)