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
def test_flushed(self) -> None:
    """
            Generate a warning and flush it.
            """
    warnings.warn(self.message, self.category)
    self.assertEqual(len(self.flushWarnings()), 1)