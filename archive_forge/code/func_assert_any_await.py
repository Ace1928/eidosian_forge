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
def assert_any_await(self, /, *args, **kwargs):
    """
        Assert the mock has ever been awaited with the specified arguments.
        """
    expected = self._call_matcher(_Call((args, kwargs), two=True))
    cause = expected if isinstance(expected, Exception) else None
    actual = [self._call_matcher(c) for c in self.await_args_list]
    if cause or expected not in _AnyComparer(actual):
        expected_string = self._format_mock_call_signature(args, kwargs)
        raise AssertionError('%s await not found' % expected_string) from cause