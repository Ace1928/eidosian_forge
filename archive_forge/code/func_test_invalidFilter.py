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
def test_invalidFilter(self) -> None:
    """
        If an object which is neither a function nor a method is included in the
        C{offendingFunctions} list, C{flushWarnings} raises L{ValueError}.  Such
        a call flushes no warnings.
        """
    warnings.warn('oh no')
    self.assertRaises(ValueError, self.flushWarnings, [None])
    self.assertEqual(len(self.flushWarnings()), 1)