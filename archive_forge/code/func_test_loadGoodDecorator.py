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
def test_loadGoodDecorator(self) -> None:
    """
        A decorated test method for which the decorator has set the method's
        __name__ correctly is loaded and the only name by which it goes is used.
        """
    from twisted.trial.test import sample
    suite = self.loader.loadAnything(sample.DecorationTest.test_goodDecorator, parent=sample.DecorationTest, qualName=['sample', 'DecorationTest', 'test_goodDecorator'])
    self.assertEqual(1, suite.countTestCases())
    self.assertEqual('test_goodDecorator', suite._testMethodName)