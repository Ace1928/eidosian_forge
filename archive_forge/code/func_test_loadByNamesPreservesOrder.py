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
def test_loadByNamesPreservesOrder(self) -> None:
    """
        L{TestLoader.loadByNames} preserves the order of tests provided to it.
        """
    modules = ['inheritancepackage.test_x.A.test_foo', 'twisted.trial.test.sample', 'goodpackage', 'twisted.trial.test.test_log', 'twisted.trial.test.sample.FooTest', 'package.test_module']
    suite1 = self.loader.loadByNames(modules)
    suite2 = runner.TestSuite(map(self.loader.loadByName, modules))
    self.assertEqual(testNames(suite1), testNames(suite2))