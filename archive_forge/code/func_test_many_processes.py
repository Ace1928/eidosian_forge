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
def test_many_processes(self):
    if self.TYPE == 'threads':
        self.skipTest('test not appropriate for {}'.format(self.TYPE))
    sm = multiprocessing.get_start_method()
    travis = os.environ.get('COVERAGE')
    N = (1 if travis else 5) if sm == 'spawn' else 100
    procs = [self.Process(target=self._test_sleep, args=(0.01,)) for i in range(N)]
    for p in procs:
        p.start()
    for p in procs:
        join_process(p)
    for p in procs:
        self.assertEqual(p.exitcode, 0)
    procs = [self.Process(target=self._sleep_some) for i in range(N)]
    for p in procs:
        p.start()
    time.sleep(0.001)
    for p in procs:
        p.terminate()
    for p in procs:
        join_process(p)
    if os.name != 'nt':
        exitcodes = [-signal.SIGTERM]
        if sys.platform == 'darwin':
            exitcodes.append(-signal.SIGKILL)
        for p in procs:
            self.assertIn(p.exitcode, exitcodes)