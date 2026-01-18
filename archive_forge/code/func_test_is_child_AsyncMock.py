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
def test_is_child_AsyncMock(self):
    mock = MagicMock(spec_set=AsyncClass)
    self.assertTrue(iscoroutinefunction(mock.async_method))
    self.assertFalse(iscoroutinefunction(mock.normal_method))
    self.assertIsInstance(mock.async_method, AsyncMock)
    self.assertIsInstance(mock.normal_method, MagicMock)
    self.assertIsInstance(mock, MagicMock)