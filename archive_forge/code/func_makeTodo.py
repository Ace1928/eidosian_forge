import inspect
import os
import sys
import tempfile
import types
import unittest as pyunit
import warnings
from dis import findlinestarts as _findlinestarts
from typing import (
from unittest import SkipTest
from attrs import frozen
from typing_extensions import ParamSpec
from twisted.internet.defer import Deferred, ensureDeferred
from twisted.python import failure, log, monkey
from twisted.python.deprecate import (
from twisted.python.reflect import fullyQualifiedName
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import itrial, util
def makeTodo(value: Union[str, Tuple[Union[Type[BaseException], Iterable[Type[BaseException]]], str]]) -> Todo:
    """
    Return a L{Todo} object built from C{value}.

    If C{value} is a string, return a Todo that expects any exception with
    C{value} as a reason. If C{value} is a tuple, the second element is used
    as the reason and the first element as the excepted error(s).

    @param value: A string or a tuple of C{(errors, reason)}, where C{errors}
    is either a single exception class or an iterable of exception classes.

    @return: A L{Todo} object.
    """
    if isinstance(value, str):
        return Todo(reason=value)
    if isinstance(value, tuple):
        errors, reason = value
        if isinstance(errors, type):
            iterableErrors: Iterable[Type[BaseException]] = [errors]
        else:
            iterableErrors = errors
        return Todo(reason=reason, errors=iterableErrors)