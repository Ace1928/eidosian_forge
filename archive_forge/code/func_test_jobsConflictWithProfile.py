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
def test_jobsConflictWithProfile(self) -> None:
    """
        C{parseOptions} raises a C{UsageError} when C{--profile} is passed
        along C{--jobs} as it's not supported yet.

        @see: U{http://twistedmatrix.com/trac/ticket/5827}
        """
    error = self.assertRaises(UsageError, self.options.parseOptions, ['--jobs', '4', '--profile'])
    self.assertEqual("You can't specify --profile when using --jobs", str(error))