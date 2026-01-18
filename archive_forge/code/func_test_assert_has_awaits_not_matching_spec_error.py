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
def test_assert_has_awaits_not_matching_spec_error(self):

    async def f(x=None):
        pass
    self.mock = AsyncMock(spec=f)
    run(self._runnable_test(1))
    with self.assertRaisesRegex(AssertionError, '^{}$'.format(re.escape('Awaits not found.\nExpected: [call()]\nActual: [call(1)]'))) as cm:
        self.mock.assert_has_awaits([call()])
    self.assertIsNone(cm.exception.__cause__)
    with self.assertRaisesRegex(AssertionError, '^{}$'.format(re.escape("Error processing expected awaits.\nErrors: [None, TypeError('too many positional arguments')]\nExpected: [call(), call(1, 2)]\nActual: [call(1)]"))) as cm:
        self.mock.assert_has_awaits([call(), call(1, 2)])
    self.assertIsInstance(cm.exception.__cause__, TypeError)