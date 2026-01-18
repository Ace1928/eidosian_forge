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
@patch(f'{async_foo_name}.async_method')
def test_no_parent_attribute(mock_method):
    m = mock_method()
    self.assertTrue(inspect.isawaitable(m))
    run(m)