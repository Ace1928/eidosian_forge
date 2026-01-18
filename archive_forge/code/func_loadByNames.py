import doctest
import importlib
import inspect
import os
import sys
import types
import unittest as pyunit
import warnings
from contextlib import contextmanager
from importlib.machinery import SourceFileLoader
from typing import Callable, Generator, List, Optional, TextIO, Type, Union
from zope.interface import implementer
from attrs import define
from typing_extensions import ParamSpec, Protocol, TypeAlias, TypeGuard
from twisted.internet import defer
from twisted.python import failure, filepath, log, modules, reflect
from twisted.trial import unittest, util
from twisted.trial._asyncrunner import _ForceGarbageCollectionDecorator, _iterateTests
from twisted.trial._synctest import _logObserver
from twisted.trial.itrial import ITestCase
from twisted.trial.reporter import UncleanWarningsReporterWrapper, _ExitWrapper
from twisted.trial.unittest import TestSuite
from . import itrial
def loadByNames(self, names: List[str], recurse: bool=False) -> TestSuite:
    """
        Load some tests by a list of names.

        @param names: A L{list} of qualified names.
        @param recurse: A boolean. If True, inspect modules within packages
            within the given package (and so on), otherwise, only inspect
            modules in the package itself.
        """
    things = []
    errors = []
    for name in names:
        try:
            things.append(self.loadByName(name, recurse=recurse))
        except BaseException:
            errors.append(ErrorHolder(name, failure.Failure()))
    things.extend(errors)
    return self.suiteFactory(self._uniqueTests(things))