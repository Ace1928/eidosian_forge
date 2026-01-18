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
def test_dryRunWithJobs(self) -> None:
    """
        L{_makeRunner} returns a L{TrialRunner} instance in C{DRY_RUN} mode
        when the C{--dry-run} option is passed, even if C{--jobs} is set.
        """
    self.options.parseOptions(['--jobs', '4', '--dry-run'])
    runner = trial._makeRunner(self.options)
    assert isinstance(runner, TrialRunner)
    self.assertIsInstance(runner, TrialRunner)
    self.assertEqual(TrialRunner.DRY_RUN, runner.mode)