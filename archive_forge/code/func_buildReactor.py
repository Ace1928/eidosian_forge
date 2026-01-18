import os
import signal
import time
from typing import TYPE_CHECKING, Callable, Dict, Optional, Sequence, Type, Union, cast
from zope.interface import Interface
from twisted.python import log
from twisted.python.deprecate import _fullyQualifiedName as fullyQualifiedName
from twisted.python.failure import Failure
from twisted.python.reflect import namedAny
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest, SynchronousTestCase
from twisted.trial.util import DEFAULT_TIMEOUT_DURATION, acquireAttribute
def buildReactor(self):
    """
        Create and return a reactor using C{self.reactorFactory}.
        """
    try:
        from twisted.internet import reactor as globalReactor
        from twisted.internet.cfreactor import CFReactor
    except ImportError:
        pass
    else:
        if isinstance(globalReactor, CFReactor) and self.reactorFactory is CFReactor:
            raise SkipTest("CFReactor uses APIs which manipulate global state, so it's not safe to run its own reactor-builder tests under itself")
    try:
        assert self.reactorFactory is not None
        reactor = self.reactorFactory()
        reactor._originalReactorDict = globalReactor.__dict__
        reactor._originalReactorClass = globalReactor.__class__
        globalReactor.__dict__ = reactor.__dict__
        globalReactor.__class__ = reactor.__class__
    except BaseException:
        log.err(None, 'Failed to install reactor')
        self.flushLoggedErrors()
        raise SkipTest(Failure().getErrorMessage())
    else:
        if self.requiredInterfaces is not None:
            missing = [required for required in self.requiredInterfaces if not required.providedBy(reactor)]
            if missing:
                self._unbuildReactor(reactor)
                raise SkipTest('%s does not provide %s' % (fullyQualifiedName(reactor.__class__), ','.join([fullyQualifiedName(x) for x in missing])))
    self.addCleanup(self._unbuildReactor, reactor)
    return reactor