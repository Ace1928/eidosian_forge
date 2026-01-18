import asyncio
import errno
import signal
from sys import version_info as py_version_info
from pexpect import EOF
def set_expecter(self, expecter):
    self.expecter = expecter
    self.fut = asyncio.Future()