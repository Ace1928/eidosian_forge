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
@classmethod
def makeTestCaseClasses(cls: Type['ReactorBuilder']) -> Dict[str, Union[Type['ReactorBuilder'], Type[SynchronousTestCase]]]:
    """
        Create a L{SynchronousTestCase} subclass which mixes in C{cls} for each
        known reactor and return a dict mapping their names to them.
        """
    classes: Dict[str, Union[Type['ReactorBuilder'], Type[SynchronousTestCase]]] = {}
    for reactor in cls._reactors:
        shortReactorName = reactor.split('.')[-1]
        name = (cls.__name__ + '.' + shortReactorName + 'Tests').replace('.', '_')

        class testcase(cls, SynchronousTestCase):
            __module__ = cls.__module__
            if reactor in cls.skippedReactors:
                skip = cls.skippedReactors[reactor]
            try:
                reactorFactory = namedAny(reactor)
            except BaseException:
                skip = Failure().getErrorMessage()
        testcase.__name__ = name
        testcase.__qualname__ = '.'.join(cls.__qualname__.split()[0:-1] + [name])
        classes[testcase.__name__] = testcase
    return classes