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
def deprecationDecorator(function):
    warningString = getDeprecationWarningString(function, version, None, replacement)

    @wraps(function)
    def deprecatedFunction(*args, **kwargs):
        warn(warningString, DeprecationWarning, stacklevel=2)
        return function(*args, **kwargs)
    _appendToDocstring(deprecatedFunction, _getDeprecationDocstring(version, replacement))
    deprecatedFunction.deprecatedVersion = version
    result = _DeprecatedProperty(deprecatedFunction)
    result.warningString = warningString
    return result