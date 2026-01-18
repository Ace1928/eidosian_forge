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
def test_loggingAnObjectWithBroken__str__(self) -> None:
    self.lp.msg(EvilStr())
    self.assertEqual(len(self.out), 1)
    self.assertNotIn(self.ERROR_UNFORMATTABLE_OBJECT, self.out[0])