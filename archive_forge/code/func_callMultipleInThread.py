from __future__ import annotations
import queue as Queue
from typing import Callable, TypeVar
from typing_extensions import ParamSpec
from twisted.internet import defer
from twisted.internet.interfaces import IReactorFromThreads
from twisted.python import failure
from twisted.python.threadpool import ThreadPool
def callMultipleInThread(tupleList):
    """
    Run a list of functions in the same thread.

    tupleList should be a list of (function, argsList, kwargsDict) tuples.
    """
    from twisted.internet import reactor
    reactor.callInThread(_runMultiple, tupleList)