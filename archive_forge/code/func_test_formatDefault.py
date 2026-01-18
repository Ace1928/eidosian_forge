import errno
import os
import re
import sys
from inspect import getmro
from io import BytesIO, StringIO
from typing import Type
from unittest import (
from hamcrest import assert_that, equal_to, has_item, has_length
from twisted.python import log
from twisted.python.failure import Failure
from twisted.trial import itrial, reporter, runner, unittest, util
from twisted.trial.reporter import UncleanWarningsReporterWrapper, _ExitWrapper
from twisted.trial.test import erroneous, sample
from twisted.trial.unittest import SkipTest, Todo, makeTodo
from .._dist.test.matchers import isFailure, matches_result, similarFrame
from .matchers import after
def test_formatDefault(self):
    tb = self.result._formatFailureTraceback(self.f)
    self.stringComparison(['Traceback (most recent call last):', '  File "foo/bar.py", line 5, in foo', re.compile('^\\s*$'), '  File "foo/bar.py", line 10, in qux', re.compile('^\\s*$'), 'RuntimeError: foo'], tb.splitlines())