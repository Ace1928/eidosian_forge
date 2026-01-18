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
def test_loadByNamesDuplicate(self) -> None:
    """
        Check that loadByNames ignores duplicate names
        """
    module = 'twisted.trial.test.test_log'
    suite1 = self.loader.loadByNames([module, module], True)
    suite2 = self.loader.loadByName(module, True)
    self.assertSuitesEqual(suite1, suite2)