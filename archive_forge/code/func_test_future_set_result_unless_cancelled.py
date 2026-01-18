from concurrent import futures
import logging
import re
import socket
import typing
import unittest
from tornado.concurrent import (
from tornado.escape import utf8, to_unicode
from tornado import gen
from tornado.iostream import IOStream
from tornado.tcpserver import TCPServer
from tornado.testing import AsyncTestCase, bind_unused_port, gen_test
def test_future_set_result_unless_cancelled(self):
    fut = Future()
    future_set_result_unless_cancelled(fut, 42)
    self.assertEqual(fut.result(), 42)
    self.assertFalse(fut.cancelled())
    fut = Future()
    fut.cancel()
    is_cancelled = fut.cancelled()
    future_set_result_unless_cancelled(fut, 42)
    self.assertEqual(fut.cancelled(), is_cancelled)
    if not is_cancelled:
        self.assertEqual(fut.result(), 42)