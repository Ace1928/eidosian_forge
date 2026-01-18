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
@deprecated(Version('Twisted', 15, 0, 0), 'twisted.internet.defer.inlineCallbacks')
def deferredGenerator(f: Callable[..., _DeferableGenerator]) -> Callable[..., Deferred[object]]:
    """
    L{deferredGenerator} and L{waitForDeferred} help you write
    L{Deferred}-using code that looks like a regular sequential function.
    Consider the use of L{inlineCallbacks} instead, which can accomplish
    the same thing in a more concise manner.

    There are two important functions involved: L{waitForDeferred}, and
    L{deferredGenerator}.  They are used together, like this::

        @deferredGenerator
        def thingummy():
            thing = waitForDeferred(makeSomeRequestResultingInDeferred())
            yield thing
            thing = thing.getResult()
            print(thing) #the result! hoorj!

    L{waitForDeferred} returns something that you should immediately yield; when
    your generator is resumed, calling C{thing.getResult()} will either give you
    the result of the L{Deferred} if it was a success, or raise an exception if it
    was a failure.  Calling C{getResult} is B{absolutely mandatory}.  If you do
    not call it, I{your program will not work}.

    L{deferredGenerator} takes one of these waitForDeferred-using generator
    functions and converts it into a function that returns a L{Deferred}. The
    result of the L{Deferred} will be the last value that your generator yielded
    unless the last value is a L{waitForDeferred} instance, in which case the
    result will be L{None}.  If the function raises an unhandled exception, the
    L{Deferred} will errback instead.  Remember that C{return result} won't work;
    use C{yield result; return} in place of that.

    Note that not yielding anything from your generator will make the L{Deferred}
    result in L{None}. Yielding a L{Deferred} from your generator is also an error
    condition; always yield C{waitForDeferred(d)} instead.

    The L{Deferred} returned from your deferred generator may also errback if your
    generator raised an exception.  For example::

        @deferredGenerator
        def thingummy():
            thing = waitForDeferred(makeSomeRequestResultingInDeferred())
            yield thing
            thing = thing.getResult()
            if thing == 'I love Twisted':
                # will become the result of the Deferred
                yield 'TWISTED IS GREAT!'
                return
            else:
                # will trigger an errback
                raise Exception('DESTROY ALL LIFE')

    Put succinctly, these functions connect deferred-using code with this 'fake
    blocking' style in both directions: L{waitForDeferred} converts from a
    L{Deferred} to the 'blocking' style, and L{deferredGenerator} converts from the
    'blocking' style to a L{Deferred}.
    """

    @wraps(f)
    def unwindGenerator(*args: object, **kwargs: object) -> Deferred[object]:
        return _deferGenerator(f(*args, **kwargs), Deferred())
    return unwindGenerator