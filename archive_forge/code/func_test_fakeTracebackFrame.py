from __future__ import annotations
import linecache
import pdb
import re
import sys
import traceback
from dis import distb
from io import StringIO
from traceback import FrameSummary
from types import TracebackType
from typing import Any, Generator
from unittest import skipIf
from cython_test_exception_raiser import raiser
from twisted.python import failure, reflect
from twisted.trial.unittest import SynchronousTestCase
def test_fakeTracebackFrame(self) -> None:
    """
        See L{FakeAttributesTests} for more details about this test.
        """
    frame = failure._Frame(('dummyname', 'dummyfilename', 42, {}, {}), None)
    traceback_frame = failure._TracebackFrame(frame)
    self.assertEqual(traceback_frame.tb_frame, frame)
    self.assertEqual(traceback_frame.tb_lineno, 42)
    self.assertIsInstance(traceback_frame.tb_lasti, int)
    self.assertTrue(hasattr(traceback_frame, 'tb_next'))