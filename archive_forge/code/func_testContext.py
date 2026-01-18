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
def testContext(self) -> None:
    catcher = self.catcher
    log.callWithContext({'subsystem': 'not the default', 'subsubsystem': 'a', 'other': 'c'}, log.callWithContext, {'subsubsystem': 'b'}, log.msg, 'foo', other='d')
    i = catcher.pop()
    self.assertEqual(i['subsubsystem'], 'b')
    self.assertEqual(i['subsystem'], 'not the default')
    self.assertEqual(i['other'], 'd')
    self.assertEqual(i['message'][0], 'foo')