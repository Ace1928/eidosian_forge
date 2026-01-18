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
def test_noMessageNoFormat(self) -> None:
    """
        If C{"format"} is unspecified and C{"message"} is empty, return
        L{None}.
        """
    eventDict = dict(message=(), isError=0)
    text = log.textFromEventDict(eventDict)
    self.assertIsNone(text)