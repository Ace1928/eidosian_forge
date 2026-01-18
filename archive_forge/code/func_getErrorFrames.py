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
def getErrorFrames(self, test):
    """
        Run the given C{test}, make sure it fails and return the trimmed
        frames.

        @param test: The test case to run.

        @return: The C{list} of frames trimmed.
        """
    stream = StringIO()
    result = reporter.Reporter(stream)
    test.run(result)
    bads = result.failures + result.errors
    self.assertEqual(len(bads), 1)
    self.assertEqual(bads[0][0], test)
    return result._trimFrames(bads[0][1].frames)