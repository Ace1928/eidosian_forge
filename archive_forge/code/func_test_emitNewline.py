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
def test_emitNewline(self) -> None:
    """
        FileLogObserver.emit() will append a newline to its file output.
        """
    output = StringIO()
    flo = log.FileLogObserver(output)
    publisher = log.LogPublisher()
    publisher.addObserver(flo.emit)
    publisher.msg('Hello!')
    result = output.getvalue()
    suffix = 'Hello!\n'
    self.assertTrue(result.endswith(suffix), f'{result!r} does not end with {suffix!r}')