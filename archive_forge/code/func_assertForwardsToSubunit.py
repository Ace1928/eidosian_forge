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
def assertForwardsToSubunit(self, methodName, *args, **kwargs):
    """
        Assert that 'methodName' on L{SubunitReporter} forwards to the
        equivalent method on subunit.

        Checks that the return value from subunit is returned from the
        L{SubunitReporter} and that the reporter writes the same data to its
        stream as subunit does to its own.

        Assumes that the method on subunit has the same name as the method on
        L{SubunitReporter}.
        """
    stream = BytesIO()
    subunitClient = reporter.TestProtocolClient(stream)
    subunitReturn = getattr(subunitClient, methodName)(*args, **kwargs)
    subunitOutput = stream.getvalue()
    reporterReturn = getattr(self.result, methodName)(*args, **kwargs)
    self.assertEqual(subunitReturn, reporterReturn)
    self.assertEqual(subunitOutput, self.stream.getvalue())