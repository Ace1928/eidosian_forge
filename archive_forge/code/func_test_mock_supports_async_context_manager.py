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
def test_mock_supports_async_context_manager(self):

    def inner_test(mock_type):
        called = False
        cm = self.WithAsyncContextManager()
        cm_mock = mock_type(cm)

        async def use_context_manager():
            nonlocal called
            async with cm_mock as result:
                called = True
            return result
        cm_result = run(use_context_manager())
        self.assertTrue(called)
        self.assertTrue(cm_mock.__aenter__.called)
        self.assertTrue(cm_mock.__aexit__.called)
        cm_mock.__aenter__.assert_awaited()
        cm_mock.__aexit__.assert_awaited()
        self.assertIsNot(cm_mock, cm_result)
    for mock_type in [AsyncMock, MagicMock]:
        with self.subTest(f'test context manager magics with {mock_type}'):
            inner_test(mock_type)