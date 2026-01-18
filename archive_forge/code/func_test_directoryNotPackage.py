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
def test_directoryNotPackage(self) -> None:
    """
        L{runner.filenameToModule} raises a C{ValueError} when the name of an
        empty directory is passed that isn't considered a valid Python package
        because it doesn't contain a C{__init__.py} file.
        """
    emptyDir = filepath.FilePath(self.parent).child('emptyDirectory')
    emptyDir.createDirectory()
    err = self.assertRaises(ValueError, runner.filenameToModule, emptyDir.path)
    self.assertEqual(str(err), f'{emptyDir.path!r} is not a package directory')