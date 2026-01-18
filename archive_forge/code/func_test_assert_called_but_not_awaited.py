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
def test_assert_called_but_not_awaited(self):
    mock = AsyncMock(AsyncClass)
    with assertNeverAwaited(self):
        mock.async_method()
    self.assertTrue(iscoroutinefunction(mock.async_method))
    mock.async_method.assert_called()
    mock.async_method.assert_called_once()
    mock.async_method.assert_called_once_with()
    with self.assertRaises(AssertionError):
        mock.assert_awaited()
    with self.assertRaises(AssertionError):
        mock.async_method.assert_awaited()