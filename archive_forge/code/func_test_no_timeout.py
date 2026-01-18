from tornado import gen, ioloop
from tornado.httpserver import HTTPServer
from tornado.locks import Event
from tornado.testing import AsyncHTTPTestCase, AsyncTestCase, bind_unused_port, gen_test
from tornado.web import Application
import asyncio
import contextlib
import inspect
import gc
import os
import platform
import sys
import traceback
import unittest
import warnings
def test_no_timeout(self):

    @gen_test(timeout=1)
    def test(self):
        yield gen.sleep(0.1)
    test(self)
    self.finished = True