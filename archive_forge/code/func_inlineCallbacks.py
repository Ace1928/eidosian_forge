from __future__ import annotations
import inspect
import traceback
import warnings
from abc import ABC, abstractmethod
from asyncio import AbstractEventLoop, Future, iscoroutine
from contextvars import Context as _Context, copy_context as _copy_context
from enum import Enum
from functools import wraps
from sys import exc_info, implementation
from types import CoroutineType, GeneratorType, MappingProxyType, TracebackType
from typing import (
import attr
from incremental import Version
from typing_extensions import Concatenate, Literal, ParamSpec, Self
from twisted.internet.interfaces import IDelayedCall, IReactorTime
from twisted.logger import Logger
from twisted.python import lockfile
from twisted.python.compat import _PYPY, cmp, comparable
from twisted.python.deprecate import deprecated, warnAboutFunction
from twisted.python.failure import Failure, _extraneous
def inlineCallbacks(f: Callable[_P, Generator[Deferred[Any], Any, _T]]) -> Callable[_P, Deferred[_T]]:
    """
    L{inlineCallbacks} helps you write L{Deferred}-using code that looks like a
    regular sequential function. For example::

        @inlineCallbacks
        def thingummy():
            thing = yield makeSomeRequestResultingInDeferred()
            print(thing)  # the result! hoorj!

    When you call anything that results in a L{Deferred}, you can simply yield it;
    your generator will automatically be resumed when the Deferred's result is
    available. The generator will be sent the result of the L{Deferred} with the
    'send' method on generators, or if the result was a failure, 'throw'.

    Things that are not L{Deferred}s may also be yielded, and your generator
    will be resumed with the same object sent back. This means C{yield}
    performs an operation roughly equivalent to L{maybeDeferred}.

    Your inlineCallbacks-enabled generator will return a L{Deferred} object, which
    will result in the return value of the generator (or will fail with a
    failure object if your generator raises an unhandled exception). Note that
    you can't use C{return result} to return a value; use C{returnValue(result)}
    instead. Falling off the end of the generator, or simply using C{return}
    will cause the L{Deferred} to have a result of L{None}.

    Be aware that L{returnValue} will not accept a L{Deferred} as a parameter.
    If you believe the thing you'd like to return could be a L{Deferred}, do
    this::

        result = yield result
        returnValue(result)

    The L{Deferred} returned from your deferred generator may errback if your
    generator raised an exception::

        @inlineCallbacks
        def thingummy():
            thing = yield makeSomeRequestResultingInDeferred()
            if thing == 'I love Twisted':
                # will become the result of the Deferred
                returnValue('TWISTED IS GREAT!')
            else:
                # will trigger an errback
                raise Exception('DESTROY ALL LIFE')

    It is possible to use the C{return} statement instead of L{returnValue}::

        @inlineCallbacks
        def loadData(url):
            response = yield makeRequest(url)
            return json.loads(response)

    You can cancel the L{Deferred} returned from your L{inlineCallbacks}
    generator before it is fired by your generator completing (either by
    reaching its end, a C{return} statement, or by calling L{returnValue}).
    A C{CancelledError} will be raised from the C{yield}ed L{Deferred} that
    has been cancelled if that C{Deferred} does not otherwise suppress it.
    """

    @wraps(f)
    def unwindGenerator(*args: _P.args, **kwargs: _P.kwargs) -> Deferred[_T]:
        try:
            gen = f(*args, **kwargs)
        except _DefGen_Return:
            raise TypeError('inlineCallbacks requires %r to produce a generator; insteadcaught returnValue being used in a non-generator' % (f,))
        if not isinstance(gen, GeneratorType):
            raise TypeError('inlineCallbacks requires %r to produce a generator; instead got %r' % (f, gen))
        return _cancellableInlineCallbacks(gen)
    return unwindGenerator