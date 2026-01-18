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
def test_functionBoundaries(self) -> None:
    """
        Verify that warnings emitted at the very edges of a function are still
        determined to be emitted from that function.
        """

    def warner() -> None:
        warnings.warn('first line warning')
        warnings.warn('internal line warning')
        warnings.warn('last line warning')
    warner()
    self.assertEqual(len(self.flushWarnings(offendingFunctions=[warner])), 3)