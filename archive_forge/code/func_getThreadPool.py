from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
def getThreadPool() -> 'ThreadPool':
    """
        Return the threadpool used by L{IReactorInThreads.callInThread}.
        Create it first if necessary.
        """