import socket
import subprocess
import sys
import textwrap
import unittest
from tornado import gen
from tornado.iostream import IOStream
from tornado.log import app_log
from tornado.tcpserver import TCPServer
from tornado.test.util import skipIfNonUnix
from tornado.testing import AsyncTestCase, ExpectLog, bind_unused_port, gen_test
from typing import Tuple
def test_bind_start(self):
    code = textwrap.dedent('\n            import warnings\n\n            from tornado.ioloop import IOLoop\n            from tornado.process import task_id\n            from tornado.tcpserver import TCPServer\n\n            warnings.simplefilter("ignore", DeprecationWarning)\n\n            server = TCPServer()\n            server.bind(0, address=\'127.0.0.1\')\n            server.start(3)\n            IOLoop.current().run_sync(lambda: None)\n            print(task_id(), end=\'\')\n        ')
    out, err = self.run_subproc(code)
    self.assertEqual(''.join(sorted(out)), '012')
    self.assertEqual(err, '')