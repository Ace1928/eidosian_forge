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
def test_assert_has_mock_calls_on_async_mock_with_spec(self):
    a_class_mock = AsyncMock(AsyncClass)
    with assertNeverAwaited(self):
        a_class_mock.async_method()
    kalls_empty = [('', (), {})]
    self.assertEqual(a_class_mock.async_method.mock_calls, kalls_empty)
    self.assertEqual(a_class_mock.mock_calls, [call.async_method()])
    with assertNeverAwaited(self):
        a_class_mock.async_method(1, 2, 3, a=4, b=5)
    method_kalls = [call(), call(1, 2, 3, a=4, b=5)]
    mock_kalls = [call.async_method(), call.async_method(1, 2, 3, a=4, b=5)]
    self.assertEqual(a_class_mock.async_method.mock_calls, method_kalls)
    self.assertEqual(a_class_mock.mock_calls, mock_kalls)