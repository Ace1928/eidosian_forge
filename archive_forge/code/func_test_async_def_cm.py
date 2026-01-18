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
def test_async_def_cm(self):

    async def test_async():
        with patch(f'{__name__}.async_func', AsyncMock()):
            self.assertIsInstance(async_func, AsyncMock)
        self.assertTrue(inspect.iscoroutinefunction(async_func))
    run(test_async())