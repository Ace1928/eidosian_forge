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
def filenameToModule(fn):
    """
    Given a filename, do whatever possible to return a module object matching
    that file.

    If the file in question is a module in Python path, properly import and
    return that module. Otherwise, load the source manually.

    @param fn: A filename.
    @return: A module object.
    @raise ValueError: If C{fn} does not exist.
    """
    oldFn = fn
    if (3, 8) <= sys.version_info < (3, 10) and (not os.path.isabs(fn)):
        fn = os.path.join(os.getcwd(), fn)
    if not os.path.exists(fn):
        raise ValueError(f"{oldFn!r} doesn't exist")
    moduleName = reflect.filenameToModuleName(fn)
    try:
        ret = reflect.namedAny(moduleName)
    except (ValueError, AttributeError):
        return _importFromFile(fn, moduleName=moduleName)
    if getattr(ret, '__file__', None) is None:
        return _importFromFile(fn, moduleName=moduleName)
    retFile = os.path.splitext(ret.__file__)[0] + '.py'
    same = getattr(os.path, 'samefile', samefile)
    if os.path.isfile(fn) and (not same(fn, retFile)):
        del sys.modules[ret.__name__]
        ret = _importFromFile(fn, moduleName=moduleName)
    return ret