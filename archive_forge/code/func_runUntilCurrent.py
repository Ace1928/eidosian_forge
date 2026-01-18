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
def runUntilCurrent(self) -> None:
    """
        Run all pending timed calls.
        """
    if self.threadCallQueue:
        count = 0
        total = len(self.threadCallQueue)
        for f, a, kw in self.threadCallQueue:
            try:
                f(*a, **kw)
            except BaseException:
                log.err()
            count += 1
            if count == total:
                break
        del self.threadCallQueue[:count]
        if self.threadCallQueue:
            self.wakeUp()
    self._insertNewDelayedCalls()
    now = self.seconds()
    while self._pendingTimedCalls and self._pendingTimedCalls[0].time <= now:
        call = heappop(self._pendingTimedCalls)
        if call.cancelled:
            self._cancellations -= 1
            continue
        if call.delayed_time > 0.0:
            call.activate_delay()
            heappush(self._pendingTimedCalls, call)
            continue
        try:
            call.called = 1
            call.func(*call.args, **call.kw)
        except BaseException:
            log.err()
            if call.creator is not None:
                e = '\n'
                e += ' C: previous exception occurred in ' + 'a DelayedCall created here:\n'
                e += ' C:'
                e += ''.join(call.creator).rstrip().replace('\n', '\n C:')
                e += '\n'
                log.msg(e)
    if self._cancellations > 50 and self._cancellations > len(self._pendingTimedCalls) >> 1:
        self._cancellations = 0
        self._pendingTimedCalls = [x for x in self._pendingTimedCalls if not x.cancelled]
        heapify(self._pendingTimedCalls)
    if self._justStopped:
        self._justStopped = False
        self.fireSystemEvent('shutdown')