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
def test_listen_multi_reuse_port(self):
    code = textwrap.dedent("\n            import asyncio\n            import socket\n            from tornado.netutil import bind_sockets\n            from tornado.process import task_id, fork_processes\n            from tornado.tcpserver import TCPServer\n\n            # Pick an unused port which we will be able to bind to multiple times.\n            (sock,) = bind_sockets(0, address='127.0.0.1',\n                family=socket.AF_INET, reuse_port=True)\n            port = sock.getsockname()[1]\n\n            fork_processes(3)\n\n            async def main():\n                server = TCPServer()\n                server.listen(port, address='127.0.0.1', reuse_port=True)\n            asyncio.run(main())\n            print(task_id(), end='')\n            ")
    out, err = self.run_subproc(code)
    self.assertEqual(''.join(sorted(out)), '012')
    self.assertEqual(err, '')