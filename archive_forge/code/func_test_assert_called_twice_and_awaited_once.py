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
def test_assert_called_twice_and_awaited_once(self):
    mock = AsyncMock(AsyncClass)
    coroutine = mock.async_method()
    with assertNeverAwaited(self):
        mock.async_method()
    with self.assertRaises(AssertionError):
        mock.async_method.assert_awaited()
    mock.async_method.assert_called()
    run(self._await_coroutine(coroutine))
    mock.async_method.assert_awaited()
    mock.async_method.assert_awaited_once()