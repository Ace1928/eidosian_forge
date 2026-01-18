import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
import unittest
from tornado.httpclient import HTTPClient, HTTPError
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.log import gen_log
from tornado.process import fork_processes, task_id, Subprocess
from tornado.simple_httpclient import SimpleAsyncHTTPClient
from tornado.testing import bind_unused_port, ExpectLog, AsyncTestCase, gen_test
from tornado.test.util import skipIfNonUnix
from tornado.web import RequestHandler, Application
@gen_test
def test_close_stdin(self):
    subproc = Subprocess([sys.executable, '-u', '-i'], stdin=Subprocess.STREAM, stdout=Subprocess.STREAM, stderr=subprocess.STDOUT)
    self.addCleanup(lambda: self.term_and_wait(subproc))
    yield subproc.stdout.read_until(b'>>> ')
    subproc.stdin.close()
    data = (yield subproc.stdout.read_until_close())
    self.assertEqual(data, b'\n')