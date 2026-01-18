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
def loadFile(self, fileName, recurse=False):
    """
        Load a file, and then the tests in that file.

        @param fileName: The file name to load.
        @param recurse: A boolean. If True, inspect modules within packages
            within the given package (and so on), otherwise, only inspect
            modules in the package itself.
        """
    name = reflect.filenameToModuleName(fileName)
    try:
        module = SourceFileLoader(name, fileName).load_module()
        return self.loadAnything(module, recurse=recurse)
    except OSError:
        raise ValueError(f'{fileName} is not a Python file.')