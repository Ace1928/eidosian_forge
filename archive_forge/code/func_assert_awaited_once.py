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
def assert_awaited_once(self):
    """
        Assert that the mock was awaited exactly once.
        """
    if not self.await_count == 1:
        msg = f'Expected {self._mock_name or 'mock'} to have been awaited once. Awaited {self.await_count} times.'
        raise AssertionError(msg)