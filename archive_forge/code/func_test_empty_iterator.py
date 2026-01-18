import asyncio
from concurrent import futures
import gc
import datetime
import platform
import sys
import time
import weakref
import unittest
from tornado.concurrent import Future
from tornado.log import app_log
from tornado.testing import AsyncHTTPTestCase, AsyncTestCase, ExpectLog, gen_test
from tornado.test.util import skipOnTravis, skipNotCPython
from tornado.web import Application, RequestHandler, HTTPError
from tornado import gen
import typing
@gen_test
def test_empty_iterator(self):
    g = gen.WaitIterator()
    self.assertTrue(g.done(), 'empty generator iterated')
    with self.assertRaises(ValueError):
        g = gen.WaitIterator(Future(), bar=Future())
    self.assertEqual(g.current_index, None, 'bad nil current index')
    self.assertEqual(g.current_future, None, 'bad nil current future')