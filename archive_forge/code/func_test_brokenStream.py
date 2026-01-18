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
def test_brokenStream(self):
    """
        Test that the reporter safely writes to its stream.
        """
    result = self.resultFactory(stream=BrokenStream(self.stream))
    result._writeln('Hello')
    self.assertEqual(self.stream.getvalue(), 'Hello\n')
    self.stream.truncate(0)
    self.stream.seek(0)
    result._writeln('Hello %s!', 'World')
    self.assertEqual(self.stream.getvalue(), 'Hello World!\n')