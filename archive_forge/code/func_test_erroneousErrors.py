from __future__ import annotations
import calendar
import logging
import os
import sys
import time
import warnings
from io import IOBase, StringIO
from typing import Callable, List
from zope.interface import implementer
from typing_extensions import Protocol
from twisted.logger import (
from twisted.logger.test.test_stdlib import handlerAndBytesIO
from twisted.python import failure, log
from twisted.python.log import LogPublisher
from twisted.trial import unittest
def test_erroneousErrors(self) -> None:
    """
        Exceptions raised by log observers are logged but the observer which
        raised the exception remains registered with the publisher.  These
        exceptions do not prevent the event from being sent to other observers
        registered with the publisher.
        """
    L1: list[log.EventDict] = []
    L2: list[log.EventDict] = []

    def broken(event: log.EventDict) -> None:
        1 // 0
    for observer in [L1.append, broken, L2.append]:
        log.addObserver(observer)
        self.addCleanup(log.removeObserver, observer)
    for i in range(3):
        L1[:] = []
        L2[:] = []
        log.msg("Howdy, y'all.", log_trace=[])
        excs = self.flushLoggedErrors(ZeroDivisionError)
        del self.catcher[:]
        self.assertEqual(len(excs), 1)
        self.assertEqual(len(L1), 2)
        self.assertEqual(len(L2), 2)
        self.assertEqual(L1[0]['message'], ("Howdy, y'all.",))
        self.assertEqual(L2[0]['message'], ("Howdy, y'all.",))