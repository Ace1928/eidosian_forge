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
def test_mock_aiter_and_anext_asyncmock(self):

    def inner_test(mock_type):
        instance = self.WithAsyncIterator()
        mock_instance = mock_type(instance)
        self.assertFalse(iscoroutinefunction(instance.__aiter__))
        self.assertFalse(iscoroutinefunction(mock_instance.__aiter__))
        self.assertTrue(iscoroutinefunction(instance.__anext__))
        self.assertTrue(iscoroutinefunction(mock_instance.__anext__))
    for mock_type in [AsyncMock, MagicMock]:
        with self.subTest(f'test aiter and anext corourtine with {mock_type}'):
            inner_test(mock_type)