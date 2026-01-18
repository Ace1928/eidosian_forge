import builtins
import socket  # needed only for sync-dns
import warnings
from abc import ABC, abstractmethod
from heapq import heapify, heappop, heappush
from traceback import format_stack
from types import FrameType
from typing import (
from zope.interface import classImplements, implementer
from twisted.internet import abstract, defer, error, fdesc, main, threads
from twisted.internet._resolver import (
from twisted.internet.defer import Deferred, DeferredList
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory
from twisted.python import log, reflect
from twisted.python.failure import Failure
from twisted.python.runtime import platform, seconds as runtimeSeconds
from ._signals import SignalHandling, _WithoutSignalHandling, _WithSignalHandling
from twisted.python import threadable
def fireEvent(self) -> None:
    """
        Call the triggers added to this event.
        """
    self.state = 'BEFORE'
    self.finishedBefore = []
    beforeResults: List[Deferred[object]] = []
    while self.before:
        callable, args, kwargs = self.before.pop(0)
        self.finishedBefore.append((callable, args, kwargs))
        try:
            result = callable(*args, **kwargs)
        except BaseException:
            log.err()
        else:
            if isinstance(result, Deferred):
                beforeResults.append(result)
    DeferredList(beforeResults).addCallback(self._continueFiring)