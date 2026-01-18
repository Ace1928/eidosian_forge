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
def flushErrors(self, *errorTypes):
    """
        Flush errors from the list of caught errors. If no arguments are
        specified, remove all errors. If arguments are specified, only remove
        errors of those types from the stored list.
        """
    if errorTypes:
        flushed = []
        remainder = []
        for f in self._errors:
            if f.check(*errorTypes):
                flushed.append(f)
            else:
                remainder.append(f)
        self._errors = remainder
    else:
        flushed = self._errors
        self._errors = []
    return flushed