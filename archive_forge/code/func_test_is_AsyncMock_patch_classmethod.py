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
def test_is_AsyncMock_patch_classmethod(self):

    @patch.object(AsyncClass, 'async_class_method')
    def test_async(mock_method):
        self.assertIsInstance(mock_method, AsyncMock)
    test_async()