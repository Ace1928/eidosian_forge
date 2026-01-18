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
def test_booleanSkip(self):
    """
        Tests can be skipped without specifying a reason by setting the 'skip'
        attribute to True. When this happens, the test output includes 'True'
        as the reason.
        """
    self.result.addSkip(self.test, True)
    self.result.done()
    output = self.stream.getvalue().splitlines()[3]
    self.assertEqual(output, 'True')