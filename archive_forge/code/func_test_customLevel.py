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
def test_customLevel(self) -> None:
    """
        Test the logLevel keyword for customizing level used.
        """
    self.lp.msg('Spam egg.', logLevel=logging.ERROR)
    self.assertIn(b'Spam egg.', self.out.getvalue())
    self.assertIn(b'ERROR', self.out.getvalue())
    self.out.seek(0, 0)
    self.out.truncate()
    self.lp.msg('Foo bar.', logLevel=logging.WARNING)
    self.assertIn(b'Foo bar.', self.out.getvalue())
    self.assertIn(b'WARNING', self.out.getvalue())