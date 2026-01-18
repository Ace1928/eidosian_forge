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
def test_is_async_AsyncMock(self):
    mock = AsyncMock(spec_set=AsyncClass.async_method)
    self.assertTrue(iscoroutinefunction(mock))
    self.assertIsInstance(mock, AsyncMock)