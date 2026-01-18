import asyncio
import gc
import inspect
import re
import unittest
from contextlib import contextmanager
from test import support
from asyncio import run, iscoroutinefunction
from unittest import IsolatedAsyncioTestCase
from unittest.mock import (ANY, call, AsyncMock, patch, MagicMock, Mock,
def test_async_arg_lists(self):

    def assert_attrs(mock):
        names = ('call_args_list', 'method_calls', 'mock_calls')
        for name in names:
            attr = getattr(mock, name)
            self.assertIsInstance(attr, _CallList)
            self.assertIsInstance(attr, list)
            self.assertEqual(attr, [])
    assert_attrs(self.mock)
    with assertNeverAwaited(self):
        self.mock()
    with assertNeverAwaited(self):
        self.mock(1, 2)
    with assertNeverAwaited(self):
        self.mock(a=3)
    self.mock.reset_mock()
    assert_attrs(self.mock)
    a_mock = AsyncMock(AsyncClass)
    with assertNeverAwaited(self):
        a_mock.async_method()
    with assertNeverAwaited(self):
        a_mock.async_method(1, a=3)
    a_mock.reset_mock()
    assert_attrs(a_mock)