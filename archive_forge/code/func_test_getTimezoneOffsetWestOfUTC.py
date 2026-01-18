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
def test_getTimezoneOffsetWestOfUTC(self) -> None:
    """
        Attempt to verify that L{FileLogObserver.getTimezoneOffset} returns
        correct values for the current C{TZ} environment setting for at least
        some cases.  This test method exercises a timezone that is west of UTC
        (and should produce positive results).
        """
    self._getTimezoneOffsetTest('America/New_York', 14400, 18000)