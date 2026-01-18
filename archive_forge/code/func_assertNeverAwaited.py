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
@contextmanager
def assertNeverAwaited(test):
    with test.assertWarnsRegex(RuntimeWarning, 'was never awaited$'):
        yield
        gc.collect()