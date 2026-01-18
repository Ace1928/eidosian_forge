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
def test_add_sockets(self):
    code = textwrap.dedent("\n            import asyncio\n            from tornado.netutil import bind_sockets\n            from tornado.process import fork_processes, task_id\n            from tornado.ioloop import IOLoop\n            from tornado.tcpserver import TCPServer\n\n            sockets = bind_sockets(0, address='127.0.0.1')\n            fork_processes(3)\n            async def post_fork_main():\n                server = TCPServer()\n                server.add_sockets(sockets)\n            asyncio.run(post_fork_main())\n            print(task_id(), end='')\n        ")
    out, err = self.run_subproc(code)
    self.assertEqual(''.join(sorted(out)), '012')
    self.assertEqual(err, '')