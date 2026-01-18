from __future__ import annotations
import inspect
import sys
from dis import findlinestarts
from functools import wraps
from types import ModuleType
from typing import Any, Callable, Dict, Optional, TypeVar, cast
from warnings import warn, warn_explicit
from incremental import Version, getVersionString
from typing_extensions import ParamSpec
def getDeprecationWarningString(callableThing, version, format=None, replacement=None):
    """
    Return a string indicating that the callable was deprecated in the given
    version.

    @type callableThing: C{callable}
    @param callableThing: Callable object to be deprecated

    @type version: L{incremental.Version}
    @param version: Version that C{callableThing} was deprecated in.

    @type format: C{str}
    @param format: A user-provided format to interpolate warning values into,
        or L{DEPRECATION_WARNING_FORMAT
        <twisted.python.deprecate.DEPRECATION_WARNING_FORMAT>} if L{None} is
        given

    @param replacement: what should be used in place of the callable. Either
        pass in a string, which will be inserted into the warning message,
        or a callable, which will be expanded to its full import path.
    @type replacement: C{str} or callable

    @return: A string describing the deprecation.
    @rtype: C{str}
    """
    return _getDeprecationWarningString(_fullyQualifiedName(callableThing), version, format, replacement)