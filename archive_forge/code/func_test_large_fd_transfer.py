import unittest
import unittest.mock
import queue as pyqueue
import textwrap
import time
import io
import itertools
import sys
import os
import gc
import errno
import signal
import array
import socket
import random
import logging
import subprocess
import struct
import operator
import pickle #XXX: use dill?
import weakref
import warnings
import test.support
import test.support.script_helper
from test import support
from test.support import hashlib_helper
from test.support import import_helper
from test.support import os_helper
from test.support import socket_helper
from test.support import threading_helper
from test.support import warnings_helper
import_helper.import_module('multiprocess.synchronize')
import threading
import multiprocess as multiprocessing
import multiprocess.connection
import multiprocess.dummy
import multiprocess.heap
import multiprocess.managers
import multiprocess.pool
import multiprocess.queues
from multiprocess import util
from multiprocess.connection import wait
from multiprocess.managers import BaseManager, BaseProxy, RemoteError
@unittest.skipUnless(HAS_REDUCTION, 'test needs multiprocessing.reduction')
@unittest.skipIf(sys.platform == 'win32', "test semantics don't make sense on Windows")
@unittest.skipIf(MAXFD <= 256, 'largest assignable fd number is too small')
@unittest.skipUnless(hasattr(os, 'dup2'), 'test needs os.dup2()')
def test_large_fd_transfer(self):
    if self.TYPE != 'processes':
        self.skipTest('only makes sense with processes')
    conn, child_conn = self.Pipe(duplex=True)
    p = self.Process(target=self._writefd, args=(child_conn, b'bar', True))
    p.daemon = True
    p.start()
    self.addCleanup(os_helper.unlink, os_helper.TESTFN)
    with open(os_helper.TESTFN, 'wb') as f:
        fd = f.fileno()
        for newfd in range(256, MAXFD):
            if not self._is_fd_assigned(newfd):
                break
        else:
            self.fail('could not find an unassigned large file descriptor')
        os.dup2(fd, newfd)
        try:
            reduction.send_handle(conn, newfd, p.pid)
        finally:
            os.close(newfd)
    p.join()
    with open(os_helper.TESTFN, 'rb') as f:
        self.assertEqual(f.read(), b'bar')