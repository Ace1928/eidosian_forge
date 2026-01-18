from __future__ import annotations
import contextvars
import functools
import gc
import re
import traceback
import types
import unittest as pyunit
import warnings
import weakref
from asyncio import (
from typing import (
from hamcrest import assert_that, empty, equal_to
from hypothesis import given
from hypothesis.strategies import integers
from typing_extensions import assert_type
from twisted.internet import defer, reactor
from twisted.internet.defer import (
from twisted.internet.task import Clock
from twisted.python import log
from twisted.python.compat import _PYPY
from twisted.python.failure import Failure
from twisted.trial import unittest
def testQueue(self) -> None:
    N, M = (2, 2)
    queue: DeferredQueue[int] = DeferredQueue(N, M)
    gotten: List[int] = []
    for i in range(M):
        queue.get().addCallback(gotten.append)
    self.assertRaises(defer.QueueUnderflow, queue.get)
    for i in range(M):
        queue.put(i)
        self.assertEqual(gotten, list(range(i + 1)))
    for i in range(N):
        queue.put(N + i)
        self.assertEqual(gotten, list(range(M)))
    self.assertRaises(defer.QueueOverflow, queue.put, None)
    gotten = []
    for i in range(N):
        queue.get().addCallback(gotten.append)
        self.assertEqual(gotten, list(range(N, N + i + 1)))
    queue = DeferredQueue()
    gotten = []
    for i in range(N):
        queue.get().addCallback(gotten.append)
    for i in range(N):
        queue.put(i)
    self.assertEqual(gotten, list(range(N)))
    queue = DeferredQueue(size=0)
    self.assertRaises(defer.QueueOverflow, queue.put, None)
    queue = DeferredQueue(backlog=0)
    self.assertRaises(defer.QueueUnderflow, queue.get)