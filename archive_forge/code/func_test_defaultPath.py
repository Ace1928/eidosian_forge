from __future__ import annotations
from typing_extensions import Protocol
from twisted.internet.defer import gatherResults
from twisted.names.dns import (
from twisted.names.hosts import Resolver, searchFileFor, searchFileForAll
from twisted.python.filepath import FilePath
from twisted.trial.unittest import SynchronousTestCase
def test_defaultPath(self) -> None:
    """
        The default hosts file used by L{Resolver} is I{/etc/hosts} if no value
        is given for the C{file} initializer parameter.
        """
    resolver = Resolver()
    self.assertEqual(b'/etc/hosts', resolver.file)