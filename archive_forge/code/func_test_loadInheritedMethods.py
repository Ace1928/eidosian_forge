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
def test_loadInheritedMethods(self) -> None:
    """
        Check that test methods names which are inherited from are all
        loaded rather than just one.
        """
    methods = ['inheritancepackage.test_x.A.test_foo', 'inheritancepackage.test_x.B.test_foo']
    suite1 = self.loader.loadByNames(methods)
    suite2 = runner.TestSuite(map(self.loader.loadByName, methods))
    self.assertSuitesEqual(suite1, suite2)