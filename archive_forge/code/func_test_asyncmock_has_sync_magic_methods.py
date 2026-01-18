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
def test_asyncmock_has_sync_magic_methods(self):
    a_mock = AsyncMock()
    self.assertTrue(hasattr(a_mock, '__enter__'))
    self.assertTrue(hasattr(a_mock, '__exit__'))
    self.assertTrue(hasattr(a_mock, '__next__'))
    self.assertTrue(hasattr(a_mock, '__len__'))