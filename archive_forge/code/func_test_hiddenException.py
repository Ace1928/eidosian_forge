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
def test_hiddenException(self) -> None:
    """
        When a function scheduled using L{IReactorTime.callLater} in a
        test method raises an exception that exception is added to the test
        result as an error.

        This happens even if the test also fails and the test failure is also
        added to the test result as a failure.

        Only really necessary for testing the deprecated style of tests that
        use iterate() directly. See
        L{erroneous.DelayedCall.testHiddenException} for more details.
        """
    test = erroneous.DelayedCall('testHiddenException')
    result = self.getResult(PyUnitTestSuite([test]))
    assert_that(result, matches_result(errors=has_length(1), failures=has_length(1)))
    [(actualCase, error)] = result.errors
    assert_that(test, equal_to(actualCase))
    assert_that(error, isFailure(type=equal_to(RuntimeError), value=after(str, equal_to('something blew up')), frames=has_item(similarFrame('go', 'erroneous.py'))))
    [(actualCase, failure)] = result.failures
    assert_that(test, equal_to(actualCase))
    assert_that(failure, isFailure(type=equal_to(test.failureException), value=after(str, equal_to('Deliberate failure to mask the hidden exception')), frames=has_item(similarFrame('testHiddenException', 'erroneous.py'))))