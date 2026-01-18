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
def failureResultOf(self, deferred, *expectedExceptionTypes):
    """
        Return the current failure result of C{deferred} or raise
        C{self.failureException}.

        @param deferred: A L{Deferred<twisted.internet.defer.Deferred>} which
            has a failure result.  This means
            L{Deferred.callback<twisted.internet.defer.Deferred.callback>} or
            L{Deferred.errback<twisted.internet.defer.Deferred.errback>} has
            been called on it and it has reached the end of its callback chain
            and the last callback or errback raised an exception or returned a
            L{failure.Failure}.
        @type deferred: L{Deferred<twisted.internet.defer.Deferred>}

        @param expectedExceptionTypes: Exception types to expect - if
            provided, and the exception wrapped by the failure result is
            not one of the types provided, then this test will fail.

        @raise SynchronousTestCase.failureException: If the
            L{Deferred<twisted.internet.defer.Deferred>} has no result, has a
            success result, or has an unexpected failure result.

        @return: The failure result of C{deferred}.
        @rtype: L{failure.Failure}
        """
    deferred = ensureDeferred(deferred)
    result = []
    deferred.addBoth(result.append)
    if not result:
        self.fail('Failure result expected on {!r}, found no result instead'.format(deferred))
    result = result[0]
    if not isinstance(result, failure.Failure):
        self.fail('Failure result expected on {!r}, found success result ({!r}) instead'.format(deferred, result))
    if expectedExceptionTypes and (not result.check(*expectedExceptionTypes)):
        expectedString = ' or '.join(['.'.join((t.__module__, t.__name__)) for t in expectedExceptionTypes])
        self.fail('Failure of type ({}) expected on {!r}, found type {!r} instead: {}'.format(expectedString, deferred, result.type, result.getTraceback()))
    return result