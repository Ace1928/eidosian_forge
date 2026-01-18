import asyncio
from datetime import timedelta
import typing  # noqa: F401
import unittest
from tornado import gen, locks
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
def test_context_manager_misuse(self):
    with self.assertRaises(RuntimeError):
        with locks.Lock():
            pass