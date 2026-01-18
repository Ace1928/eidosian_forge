from __future__ import annotations
import contextvars
import functools
import gc
import re
import traceback
import types
import unittest as pyunit
import warnings
import weakref
from asyncio import (
from typing import (
from hamcrest import assert_that, empty, equal_to
from hypothesis import given
from hypothesis.strategies import integers
from typing_extensions import assert_type
from twisted.internet import defer, reactor
from twisted.internet.defer import (
from twisted.internet.task import Clock
from twisted.python import log
from twisted.python.compat import _PYPY
from twisted.python.failure import Failure
from twisted.trial import unittest
def callAllSoonCalls(loop: AbstractEventLoop) -> None:
    """
    Tickle an asyncio event loop to call all of the things scheduled with
    call_soon, inasmuch as this can be done via the public API.

    @param loop: The asyncio event loop to flush the previously-called
        C{call_soon} entries from.
    """
    loop.call_soon(loop.stop)
    loop.run_forever()