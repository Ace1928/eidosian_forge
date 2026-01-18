from __future__ import annotations
import gc
import re
import sys
import textwrap
import types
from io import StringIO
from typing import List
from hamcrest import assert_that, contains_string
from hypothesis import given
from hypothesis.strategies import sampled_from
from twisted.logger import Logger
from twisted.python import util
from twisted.python.filepath import FilePath, IFilePath
from twisted.python.usage import UsageError
from twisted.scripts import trial
from twisted.trial import unittest
from twisted.trial._dist.disttrial import DistTrialRunner
from twisted.trial._dist.functional import compose
from twisted.trial.runner import (
from twisted.trial.test.test_loader import testNames
from .matchers import fileContents
def test_jobs(self) -> None:
    """
        L{_makeRunner} returns a L{DistTrialRunner} instance when the C{--jobs}
        option is passed.  The L{DistTrialRunner} knows how many workers to
        run and the C{workerArguments} to pass to them.
        """
    self.options.parseOptions(['--jobs', '4', '--force-gc'])
    runner = trial._makeRunner(self.options)
    assert isinstance(runner, DistTrialRunner)
    self.assertIsInstance(runner, DistTrialRunner)
    self.assertEqual(4, runner._maxWorkers)
    self.assertEqual(['--force-gc'], runner._workerArguments)