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
def test_alphabeticalPackage(self) -> None:
    """
        --order=alphabetical causes trial to run test modules within a given
        package alphabetically, with tests within each module alphabetized.
        """
    self.config.parseOptions(['--order', 'alphabetical', 'twisted.trial.test'])
    loader = trial._getLoader(self.config)
    suite = loader.loadByNames(self.config['tests'])
    names = testNames(suite)
    self.assertTrue(names, msg='Failed to load any tests!')
    self.assertEqual(names, sorted(names))