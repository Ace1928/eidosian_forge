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
def test_mock_async_for(self):

    async def iterate(iterator):
        accumulator = []
        async for item in iterator:
            accumulator.append(item)
        return accumulator
    expected = ['FOO', 'BAR', 'BAZ']

    def test_default(mock_type):
        mock_instance = mock_type(self.WithAsyncIterator())
        self.assertEqual(run(iterate(mock_instance)), [])

    def test_set_return_value(mock_type):
        mock_instance = mock_type(self.WithAsyncIterator())
        mock_instance.__aiter__.return_value = expected[:]
        self.assertEqual(run(iterate(mock_instance)), expected)

    def test_set_return_value_iter(mock_type):
        mock_instance = mock_type(self.WithAsyncIterator())
        mock_instance.__aiter__.return_value = iter(expected[:])
        self.assertEqual(run(iterate(mock_instance)), expected)
    for mock_type in [AsyncMock, MagicMock]:
        with self.subTest(f'default value with {mock_type}'):
            test_default(mock_type)
        with self.subTest(f'set return_value with {mock_type}'):
            test_set_return_value(mock_type)
        with self.subTest(f'set return_value iterator with {mock_type}'):
            test_set_return_value_iter(mock_type)