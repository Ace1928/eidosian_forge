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
def test_doctestError(self):
    """
        A problem encountered while running a doctest is reported in the output
        stream with a I{FAIL} or I{ERROR} tag along with a summary of what
        problem was encountered and the ID of the test.
        """
    from twisted.trial.test import erroneous
    suite = unittest.decorate(self.loader.loadDoctests(erroneous), itrial.ITestCase)
    output = self.getOutput(suite)
    path = 'twisted.trial.test.erroneous.unexpectedException'
    for substring in ['1/0', 'ZeroDivisionError', 'Exception raised:', path]:
        self.assertSubstring(substring, output)
    self.assertTrue(re.search('Fail(ed|ure in) example:', output), "Couldn't match 'Failure in example: ' or 'Failed example: '")
    expect = [self.doubleSeparator, re.compile('\\[(ERROR|FAIL)\\]')]
    self.stringComparison(expect, output.splitlines())