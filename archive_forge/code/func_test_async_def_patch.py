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
def test_async_def_patch(self):

    @patch(f'{__name__}.async_func', return_value=1)
    @patch(f'{__name__}.async_func_args', return_value=2)
    async def test_async(func_args_mock, func_mock):
        self.assertEqual(func_args_mock._mock_name, 'async_func_args')
        self.assertEqual(func_mock._mock_name, 'async_func')
        self.assertIsInstance(async_func, AsyncMock)
        self.assertIsInstance(async_func_args, AsyncMock)
        self.assertEqual(await async_func(), 1)
        self.assertEqual(await async_func_args(1, 2, c=3), 2)
    run(test_async())
    self.assertTrue(inspect.iscoroutinefunction(async_func))