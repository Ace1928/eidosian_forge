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
def test_context_manager_raise_exception_by_default(self):

    async def raise_in(context_manager):
        async with context_manager:
            raise TypeError()
    instance = self.WithAsyncContextManager()
    mock_instance = MagicMock(instance)
    with self.assertRaises(TypeError):
        run(raise_in(mock_instance))