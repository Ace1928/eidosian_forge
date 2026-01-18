from typing import Optional, Sequence, Type
from zope.interface import Interface, implementer
from twisted.internet.defer import Deferred, DeferredList
from twisted.internet.endpoints import (
from twisted.internet.error import ConnectionClosed
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.internet.test.test_tcp import (
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest
def test_badContext(self):
    """
        If the context factory passed to L{IReactorSSL.listenSSL} raises an
        exception from its C{getContext} method, that exception is raised by
        L{IReactorSSL.listenSSL}.
        """

    def useIt(reactor, contextFactory):
        return reactor.listenSSL(0, ServerFactory(), contextFactory)
    self._testBadContext(useIt)