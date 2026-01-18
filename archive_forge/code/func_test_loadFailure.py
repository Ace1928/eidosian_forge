from __future__ import annotations
import os
import sys
import unittest as pyunit
from hashlib import md5
from operator import attrgetter
from types import ModuleType
from typing import TYPE_CHECKING, Callable, Generator
from hamcrest import assert_that, equal_to, has_properties
from hamcrest.core.matcher import Matcher
from twisted.python import filepath, util
from twisted.python.modules import PythonAttribute, PythonModule, getModule
from twisted.python.reflect import ModuleNotFound
from twisted.trial import reporter, runner, unittest
from twisted.trial._asyncrunner import _iterateTests
from twisted.trial.itrial import ITestCase
from twisted.trial.test import packages
from .matchers import after
def test_loadFailure(self) -> None:
    """
        Loading a test that fails and getting the result of it ends up with one
        test ran and one failure.
        """
    suite = self.loader.loadByName('twisted.trial.test.erroneous.TestRegularFail.test_fail')
    result = reporter.TestResult()
    suite.run(result)
    self.assertEqual(result.testsRun, 1)
    self.assertEqual(len(result.failures), 1)