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
def test_finish_exception_handler(self):

    class Handler(web.RequestHandler):

        def get(self):
            raise web.Finish('ok\n')
    asyncio.run(self.run_handler(Handler))