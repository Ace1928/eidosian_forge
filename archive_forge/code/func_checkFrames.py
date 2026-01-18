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
def checkFrames(self, observedFrames, expectedFrames):
    for observed, expected in zip(observedFrames, expectedFrames):
        self.assertEqual(observed[0], expected[0])
        observedSegs = os.path.splitext(observed[1])[0].split(os.sep)
        expectedSegs = expected[1].split('/')
        self.assertEqual(observedSegs[-len(expectedSegs):], expectedSegs)
    self.assertEqual(len(observedFrames), len(expectedFrames))