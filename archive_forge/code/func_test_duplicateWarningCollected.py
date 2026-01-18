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
def test_duplicateWarningCollected(self) -> None:
    """
        Subsequent emissions of a warning from a particular source site can be
        collected by L{_collectWarnings}.  In particular, the per-module
        emitted-warning cache should be bypassed (I{__warningregistry__}).
        """
    global __warningregistry__
    del __warningregistry__

    def f() -> None:
        warnings.warn('foo')
    warnings.simplefilter('default')
    f()
    events: list[_Warning] = []
    _collectWarnings(events.append, f)
    self.assertEqual(len(events), 1)
    self.assertEqual(events[0].message, 'foo')
    self.assertEqual(len(self.flushWarnings()), 1)