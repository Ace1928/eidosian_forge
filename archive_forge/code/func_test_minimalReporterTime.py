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
def test_minimalReporterTime(self):
    """
        L{reporter.MinimalReporter} reports the time to run the tests as first
        data in its output.
        """
    times = [1.0, 1.2, 1.5, 1.9]
    result = reporter.MinimalReporter(self.stream)
    result._getTime = lambda: times.pop(0)
    self.test.run(result)
    result._printSummary()
    output = self.stream.getvalue().strip().split(' ')
    timer = output[0]
    self.assertEqual(timer, '0.7')