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
def findByName(self, _name, recurse=False):
    """
        Find and load tests, given C{name}.

        @param _name: The qualified name of the thing to load.
        @param recurse: A boolean. If True, inspect modules within packages
            within the given package (and so on), otherwise, only inspect
            modules in the package itself.

        @return: If C{name} is a filename, return the module. If C{name} is a
        fully-qualified Python name, return the object it refers to.
        """
    if os.sep in _name:
        name = reflect.filenameToModuleName(_name)
        try:
            __import__(name)
        except ImportError:
            return self.loadFile(_name, recurse=recurse)
    else:
        name = _name
    obj = parent = remaining = None
    for searchName, remainingName in _qualNameWalker(name):
        try:
            obj = reflect.namedModule(searchName)
            remaining = remainingName
            break
        except ImportError:
            tb = sys.exc_info()[2]
            while tb.tb_next is not None:
                tb = tb.tb_next
            filenameWhereHappened = tb.tb_frame.f_code.co_filename
            if filenameWhereHappened != reflect.__file__:
                raise
            if remaining == '':
                raise reflect.ModuleNotFound(f'The module {name} does not exist.')
    if obj is None:
        obj = reflect.namedAny(name)
        remaining = name.split('.')[len('.'.split(obj.__name__)) + 1:]
    try:
        for part in remaining:
            parent, obj = (obj, getattr(obj, part))
    except AttributeError:
        raise AttributeError(f'{name} does not exist.')
    return self.loadAnything(obj, parent=parent, qualName=remaining, recurse=recurse)