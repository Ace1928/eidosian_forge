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
def test_getWorkerArguments(self) -> None:
    """
        C{_getWorkerArguments} discards options like C{random} as they only
        matter in the manager, and forwards options like C{recursionlimit} or
        C{disablegc}.
        """
    self.addCleanup(sys.setrecursionlimit, sys.getrecursionlimit())
    if gc.isenabled():
        self.addCleanup(gc.enable)
    self.options.parseOptions(['--recursionlimit', '2000', '--random', '4', '--disablegc'])
    args = self.options._getWorkerArguments()
    self.assertIn('--disablegc', args)
    args.remove('--disablegc')
    self.assertEqual(['--recursionlimit', '2000'], args)