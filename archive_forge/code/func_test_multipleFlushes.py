from __future__ import annotations
import sys
import warnings
from io import StringIO
from typing import Mapping, Sequence, TypeVar
from unittest import TestResult
from twisted.python.filepath import FilePath
from twisted.trial._synctest import (
from twisted.trial.unittest import SynchronousTestCase
import warnings
import warnings
def test_multipleFlushes(self) -> None:
    """
        Any warnings emitted after a call to C{flushWarnings} can be flushed by
        another call to C{flushWarnings}.
        """
    warnings.warn('first message')
    self.assertEqual(len(self.flushWarnings()), 1)
    warnings.warn('second message')
    self.assertEqual(len(self.flushWarnings()), 1)