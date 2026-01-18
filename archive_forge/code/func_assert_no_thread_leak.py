import asyncio
import threading
import time
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor
from tornado import gen
from tornado.ioloop import IOLoop
from tornado.platform.asyncio import (
from tornado.testing import AsyncTestCase, gen_test
def assert_no_thread_leak(self):
    deadline = time.time() + 1
    while time.time() < deadline:
        threads = list(threading.enumerate())
        if len(threads) <= self.orig_thread_count:
            break
        time.sleep(0.1)
    self.assertLessEqual(len(threads), self.orig_thread_count, threads)