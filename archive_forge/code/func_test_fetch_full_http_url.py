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
def test_fetch_full_http_url(self):
    path = 'http://127.0.0.1:%d/path' % self.second_port
    response = self.fetch(path)
    self.assertEqual(response.request.url, path)