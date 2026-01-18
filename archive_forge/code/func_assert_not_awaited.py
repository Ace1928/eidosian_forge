import asyncio
import contextlib
import io
import inspect
import pprint
import sys
import builtins
import pkgutil
from asyncio import iscoroutinefunction
from types import CodeType, ModuleType, MethodType
from unittest.util import safe_repr
from functools import wraps, partial
from threading import RLock
def assert_not_awaited(self):
    """
        Assert that the mock was never awaited.
        """
    if self.await_count != 0:
        msg = f'Expected {self._mock_name or 'mock'} to not have been awaited. Awaited {self.await_count} times.'
        raise AssertionError(msg)