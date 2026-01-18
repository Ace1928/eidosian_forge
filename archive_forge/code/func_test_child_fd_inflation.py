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
def test_child_fd_inflation(self):
    if self.TYPE == 'threads':
        self.skipTest('test not appropriate for {}'.format(self.TYPE))
    sm = multiprocessing.get_start_method()
    if sm == 'fork':
        self.skipTest('test not appropriate for {}'.format(sm))
    N = 5
    evt = self.Event()
    q = self.Queue()
    procs = [self.Process(target=self._test_child_fd_inflation, args=(evt, q)) for i in range(N)]
    for p in procs:
        p.start()
    try:
        fd_counts = [q.get() for i in range(N)]
        self.assertEqual(len(set(fd_counts)), 1, fd_counts)
    finally:
        evt.set()
        for p in procs:
            p.join()
        close_queue(q)