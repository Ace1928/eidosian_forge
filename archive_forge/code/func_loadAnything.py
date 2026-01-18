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
def loadAnything(self, obj, recurse=False, parent=None, qualName=None):
    """
        Load absolutely anything (as long as that anything is a module,
        package, class, or method (with associated parent class and qualname).

        @param obj: The object to load.
        @param recurse: A boolean. If True, inspect modules within packages
            within the given package (and so on), otherwise, only inspect
            modules in the package itself.
        @param parent: If C{obj} is a method, this is the parent class of the
            method. C{qualName} is also required.
        @param qualName: If C{obj} is a method, this a list containing is the
            qualified name of the method. C{parent} is also required.

        @return: A C{TestCase} or C{TestSuite}.
        """
    if isinstance(obj, types.ModuleType):
        if isPackage(obj):
            return self.loadPackage(obj, recurse=recurse)
        return self.loadTestsFromModule(obj)
    elif isinstance(obj, type) and issubclass(obj, pyunit.TestCase):
        return self.loadTestsFromTestCase(obj)
    elif isinstance(obj, types.FunctionType) and isinstance(parent, type) and issubclass(parent, pyunit.TestCase):
        name = qualName[-1]
        inst = parent(name)
        assert getattr(inst, inst._testMethodName).__func__ == obj
        return inst
    elif isinstance(obj, TestSuite):
        return obj
    else:
        raise TypeError(f"don't know how to make test from: {obj}")