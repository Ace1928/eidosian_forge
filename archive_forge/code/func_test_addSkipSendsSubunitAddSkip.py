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
def test_addSkipSendsSubunitAddSkip(self):
    """
        Some versions of subunit have "addSkip". For these versions, when we
        call 'addSkip' on the test result, we pass the test and reason through
        to the subunit client.
        """
    addSkipCalls = []

    def addSkip(test, reason):
        addSkipCalls.append((test, reason))
    self.result._subunit.addSkip = addSkip
    self.result.addSkip(self.test, 'reason')
    self.assertEqual(addSkipCalls, [(self.test, 'reason')])