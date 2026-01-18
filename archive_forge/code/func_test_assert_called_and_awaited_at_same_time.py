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
def test_assert_called_and_awaited_at_same_time(self):
    with self.assertRaises(AssertionError):
        self.mock.assert_awaited()
    with self.assertRaises(AssertionError):
        self.mock.assert_called()
    run(self._runnable_test())
    self.mock.assert_called_once()
    self.mock.assert_awaited_once()