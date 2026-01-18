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
def test_microsecondTimestampFormatting(self) -> None:
    """
        L{FileLogObserver.formatTime} supports a value of C{timeFormat} which
        includes C{"%f"}, a L{datetime}-only format specifier for microseconds.
        """
    self.flo.timeFormat = '%f'
    self.assertEqual('600000', self.flo.formatTime(112345.6))