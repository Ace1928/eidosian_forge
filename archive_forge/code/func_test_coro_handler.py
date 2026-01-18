import asyncio
import contextlib
import gc
import io
import sys
import traceback
import types
import typing
import unittest
import tornado
from tornado import web, gen, httpclient
from tornado.test.util import skipNotCPython
def test_coro_handler(self):

    class Handler(web.RequestHandler):

        @gen.coroutine
        def get(self):
            yield asyncio.sleep(0.01)
            self.write('ok\n')
    asyncio.run(self.run_handler(Handler))