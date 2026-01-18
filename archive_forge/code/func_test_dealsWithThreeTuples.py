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
def test_dealsWithThreeTuples(self):
    """
        Some annoying stuff can pass three-tuples to addError instead of
        Failures (like PyUnit). The wrapper, of course, handles this case,
        since it is a part of L{twisted.trial.itrial.IReporter}! But it does
        not convert L{DirtyReactorAggregateError} to warnings in this case,
        because nobody should be passing those in the form of three-tuples.
        """
    result = UncleanWarningsReporterWrapper(reporter.Reporter(stream=self.output))
    result.addError(self.test, (self.dirtyError.type, self.dirtyError.value, None))
    self.assertEqual(len(result._originalReporter.errors), 1)
    self.assertEqual(result._originalReporter.errors[0][1].type, self.dirtyError.type)
    self.assertEqual(result._originalReporter.errors[0][1].value, self.dirtyError.value)