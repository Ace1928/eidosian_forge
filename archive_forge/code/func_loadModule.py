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
def loadModule(self, module):
    """
        Return a test suite with all the tests from a module.

        Included are TestCase subclasses and doctests listed in the module's
        __doctests__ module. If that's not good for you, put a function named
        either C{testSuite} or C{test_suite} in your module that returns a
        TestSuite, and I'll use the results of that instead.

        If C{testSuite} and C{test_suite} are both present, then I'll use
        C{testSuite}.
        """
    if not isinstance(module, types.ModuleType):
        raise TypeError(f'{module!r} is not a module')
    if hasattr(module, 'testSuite'):
        return module.testSuite()
    elif hasattr(module, 'test_suite'):
        return module.test_suite()
    suite = self.suiteFactory()
    for testClass in self.findTestClasses(module):
        suite.addTest(self.loadClass(testClass))
    if not hasattr(module, '__doctests__'):
        return suite
    docSuite = self.suiteFactory()
    for docTest in module.__doctests__:
        docSuite.addTest(self.loadDoctests(docTest))
    return self.suiteFactory([suite, docSuite])