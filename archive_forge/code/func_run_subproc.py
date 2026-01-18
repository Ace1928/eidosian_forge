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
def run_subproc(self, code: str) -> Tuple[str, str]:
    try:
        result = subprocess.run([sys.executable, '-Werror::DeprecationWarning'], capture_output=True, input=code, encoding='utf8', check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f'Process returned {e.returncode} stdout={e.stdout} stderr={e.stderr}') from e
    return (result.stdout, result.stderr)