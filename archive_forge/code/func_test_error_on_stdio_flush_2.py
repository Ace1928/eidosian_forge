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
def test_error_on_stdio_flush_2(self):
    for stream_name in ('stdout', 'stderr'):
        for action in ('close', 'remove'):
            old_stream = getattr(sys, stream_name)
            try:
                evt = self.Event()
                proc = self.Process(target=self._test_error_on_stdio_flush, args=(evt, {stream_name: action}))
                proc.start()
                proc.join()
                self.assertTrue(evt.is_set())
                self.assertEqual(proc.exitcode, 0)
            finally:
                setattr(sys, stream_name, old_stream)