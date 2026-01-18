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
def test_aiter_set_return_value(self):
    mock_iter = AsyncMock(name='tester')
    mock_iter.__aiter__.return_value = [1, 2, 3]

    async def main():
        return [i async for i in mock_iter]
    result = run(main())
    self.assertEqual(result, [1, 2, 3])