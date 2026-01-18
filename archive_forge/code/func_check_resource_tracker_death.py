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
def check_resource_tracker_death(self, signum, should_die):
    from multiprocess.resource_tracker import _resource_tracker
    pid = _resource_tracker._pid
    if pid is not None:
        os.kill(pid, signal.SIGKILL)
        support.wait_process(pid, exitcode=-signal.SIGKILL)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _resource_tracker.ensure_running()
    pid = _resource_tracker._pid
    os.kill(pid, signum)
    time.sleep(1.0)
    ctx = multiprocessing.get_context('spawn')
    with warnings.catch_warnings(record=True) as all_warn:
        warnings.simplefilter('always')
        sem = ctx.Semaphore()
        sem.acquire()
        sem.release()
        wr = weakref.ref(sem)
        del sem
        gc.collect()
        self.assertIsNone(wr())
        if should_die:
            self.assertEqual(len(all_warn), 1)
            the_warn = all_warn[0]
            self.assertTrue(issubclass(the_warn.category, UserWarning))
            self.assertTrue('resource_tracker: process died' in str(the_warn.message))
        else:
            self.assertEqual(len(all_warn), 0)