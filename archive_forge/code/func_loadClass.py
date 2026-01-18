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
def loadClass(self, klass):
    """
        Given a class which contains test cases, return a list of L{TestCase}s.

        @param klass: The class to load tests from.
        """
    if not isinstance(klass, type):
        raise TypeError(f'{klass!r} is not a class')
    if not isTestCase(klass):
        raise ValueError(f'{klass!r} is not a test case')
    names = self.getTestCaseNames(klass)
    tests = self.sort([self._makeCase(klass, self.methodPrefix + name) for name in names])
    return self.suiteFactory(tests)