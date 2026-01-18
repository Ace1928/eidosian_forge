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
def test_exceptionTodo(self):
    """
        The exception for expected failures should be shown in the
        C{printErrors} output.
        """
    try:
        1 / 0
    except Exception as e:
        error = e
    self.result.addExpectedFailure(self.test, Failure(error), makeTodo('todo!'))
    self.result.done()
    output = '\n'.join(self.stream.getvalue().splitlines()[3:]).strip()
    self.assertTrue(str(error) in output)