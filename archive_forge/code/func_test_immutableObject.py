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
def test_immutableObject(self) -> None:
    """
        L{_collectWarnings}'s behavior is not altered by the presence of an
        object which cannot have attributes set on it as a value in
        C{sys.modules}.
        """
    key = object()
    sys.modules[key] = key
    self.addCleanup(sys.modules.pop, key)
    self.test_duplicateWarningCollected()