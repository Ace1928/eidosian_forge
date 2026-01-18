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
def loadSortedPackages(self, sorter: Callable[[runner._Loadable], SupportsRichComparison]=runner.name) -> None:
    """
        Verify that packages are loaded in the correct order.
        """
    import uberpackage
    self.loader.sorter = sorter
    suite = self.loader.loadPackage(uberpackage, recurse=True)
    suite = unittest.decorate(suite, ITestCase)
    resultingTests = list(_iterateTests(suite))
    manifest = list(self._trialSortAlgorithm(sorter))
    for number, (manifestTest, actualTest) in enumerate(zip(manifest, resultingTests)):
        self.assertEqual(manifestTest.name, actualTest.id(), '#%d: %s != %s' % (number, manifestTest.name, actualTest.id()))
    self.assertEqual(len(manifest), len(resultingTests))