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
def test_add_thread_close_idempotent(self):
    loop = AddThreadSelectorEventLoop(asyncio.get_event_loop())
    loop.close()
    loop.close()