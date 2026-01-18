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
def test_noTodoProvided(self):
    """
        If no C{Todo} is provided to C{addExpectedFailure}, then
        L{reporter.Reporter} makes up a sensible default.

        This allows standard Python unittests to use Twisted reporters.
        """
    failure = Failure(Exception())
    self.result.addExpectedFailure(self.test, failure)
    [(test, error, todo)] = self._getTodos(self.result)
    self.assertEqual(test, self.test)
    self.assertEqual(error, failure)
    self.assertEqual(repr(todo), repr(makeTodo('Test expected to fail')))