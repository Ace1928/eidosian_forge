from __future__ import annotations
import locale
import os
import sys
from io import StringIO
from typing import Generator
from zope.interface import implementer
from hamcrest import assert_that, equal_to
from twisted.internet.base import DelayedCall
from twisted.internet.interfaces import IProcessTransport
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.trial import util
from twisted.trial.unittest import SynchronousTestCase
from twisted.trial.util import (
def test_fourWords(self) -> None:
    """
        If a delimiter is specified, it is used instead of the default comma.
        """
    sample = ['One', 'Two', 'Three', 'Four']
    expected = 'One; Two; Three; or Four'
    result = util._listToPhrase(sample, 'or', delimiter='; ')
    self.assertEqual(expected, result)