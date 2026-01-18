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
def test_async_method_calls_recorded(self):
    with assertNeverAwaited(self):
        self.mock.something(3, fish=None)
    with assertNeverAwaited(self):
        self.mock.something_else.something(6, cake=sentinel.Cake)
    self.assertEqual(self.mock.method_calls, [('something', (3,), {'fish': None}), ('something_else.something', (6,), {'cake': sentinel.Cake})], 'method calls not recorded correctly')
    self.assertEqual(self.mock.something_else.method_calls, [('something', (6,), {'cake': sentinel.Cake})], 'method calls not recorded correctly')