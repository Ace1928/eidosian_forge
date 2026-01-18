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
def test_connection(self):
    conn, child_conn = self.Pipe()
    p = self.Process(target=self._echo, args=(child_conn,))
    p.daemon = True
    p.start()
    seq = [1, 2.25, None]
    msg = latin('hello world')
    longmsg = msg * 10
    arr = array.array('i', list(range(4)))
    if self.TYPE == 'processes':
        self.assertEqual(type(conn.fileno()), int)
    self.assertEqual(conn.send(seq), None)
    self.assertEqual(conn.recv(), seq)
    self.assertEqual(conn.send_bytes(msg), None)
    self.assertEqual(conn.recv_bytes(), msg)
    if self.TYPE == 'processes':
        buffer = array.array('i', [0] * 10)
        expected = list(arr) + [0] * (10 - len(arr))
        self.assertEqual(conn.send_bytes(arr), None)
        self.assertEqual(conn.recv_bytes_into(buffer), len(arr) * buffer.itemsize)
        self.assertEqual(list(buffer), expected)
        buffer = array.array('i', [0] * 10)
        expected = [0] * 3 + list(arr) + [0] * (10 - 3 - len(arr))
        self.assertEqual(conn.send_bytes(arr), None)
        self.assertEqual(conn.recv_bytes_into(buffer, 3 * buffer.itemsize), len(arr) * buffer.itemsize)
        self.assertEqual(list(buffer), expected)
        buffer = bytearray(latin(' ' * 40))
        self.assertEqual(conn.send_bytes(longmsg), None)
        try:
            res = conn.recv_bytes_into(buffer)
        except multiprocessing.BufferTooShort as e:
            self.assertEqual(e.args, (longmsg,))
        else:
            self.fail('expected BufferTooShort, got %s' % res)
    poll = TimingWrapper(conn.poll)
    self.assertEqual(poll(), False)
    self.assertTimingAlmostEqual(poll.elapsed, 0)
    self.assertEqual(poll(-1), False)
    self.assertTimingAlmostEqual(poll.elapsed, 0)
    self.assertEqual(poll(TIMEOUT1), False)
    self.assertTimingAlmostEqual(poll.elapsed, TIMEOUT1)
    conn.send(None)
    time.sleep(0.1)
    self.assertEqual(poll(TIMEOUT1), True)
    self.assertTimingAlmostEqual(poll.elapsed, 0)
    self.assertEqual(conn.recv(), None)
    really_big_msg = latin('X') * (1024 * 1024 * 16)
    conn.send_bytes(really_big_msg)
    self.assertEqual(conn.recv_bytes(), really_big_msg)
    conn.send_bytes(SENTINEL)
    child_conn.close()
    if self.TYPE == 'processes':
        self.assertEqual(conn.readable, True)
        self.assertEqual(conn.writable, True)
        self.assertRaises(EOFError, conn.recv)
        self.assertRaises(EOFError, conn.recv_bytes)
    p.join()