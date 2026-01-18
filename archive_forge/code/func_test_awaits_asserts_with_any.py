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
def test_awaits_asserts_with_any(self):

    class Foo:

        def __eq__(self, other):
            pass
    run(self._runnable_test(Foo(), 1))
    self.mock.assert_has_awaits([call(ANY, 1)])
    self.mock.assert_awaited_with(ANY, 1)
    self.mock.assert_any_await(ANY, 1)