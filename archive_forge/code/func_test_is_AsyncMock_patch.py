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
def test_is_AsyncMock_patch(self):

    @patch.object(AsyncClass, 'async_method', spec_set=True)
    def test_async(async_method):
        self.assertIsInstance(async_method, AsyncMock)
    test_async()