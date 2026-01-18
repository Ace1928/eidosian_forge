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
def isPackageDirectory(dirname):
    """
    Is the directory at path 'dirname' a Python package directory?
    Returns the name of the __init__ file (it may have a weird extension)
    if dirname is a package directory.  Otherwise, returns False
    """

    def _getSuffixes():
        return importlib.machinery.all_suffixes()
    for ext in _getSuffixes():
        initFile = '__init__' + ext
        if os.path.exists(os.path.join(dirname, initFile)):
            return initFile
    return False