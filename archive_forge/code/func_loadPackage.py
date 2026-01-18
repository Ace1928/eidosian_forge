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
def loadPackage(self, package, recurse=False):
    """
        Load tests from a module object representing a package, and return a
        TestSuite containing those tests.

        Tests are only loaded from modules whose name begins with 'test_'
        (or whatever C{modulePrefix} is set to).

        @param package: a types.ModuleType object (or reasonable facsimile
        obtained by importing) which may contain tests.

        @param recurse: A boolean.  If True, inspect modules within packages
        within the given package (and so on), otherwise, only inspect modules
        in the package itself.

        @raise TypeError: If C{package} is not a package.

        @return: a TestSuite created with my suiteFactory, containing all the
        tests.
        """
    if not isPackage(package):
        raise TypeError(f'{package!r} is not a package')
    pkgobj = modules.getModule(package.__name__)
    if recurse:
        discovery = pkgobj.walkModules()
    else:
        discovery = pkgobj.iterModules()
    discovered = []
    for disco in discovery:
        if disco.name.split('.')[-1].startswith(self.modulePrefix):
            discovered.append(disco)
    suite = self.suiteFactory()
    for modinfo in self.sort(discovered):
        try:
            module = modinfo.load()
        except BaseException:
            thingToAdd = ErrorHolder(modinfo.name, failure.Failure())
        else:
            thingToAdd = self.loadModule(module)
        suite.addTest(thingToAdd)
    return suite