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
def test_imap_handle_iterable_exception(self):
    if self.TYPE == 'manager':
        self.skipTest('test not appropriate for {}'.format(self.TYPE))
    it = self.pool.imap(sqr, exception_throwing_generator(1, -1), 1)
    self.assertRaises(SayWhenError, it.__next__)
    it = self.pool.imap(sqr, exception_throwing_generator(1, -1), 1)
    self.assertRaises(SayWhenError, it.__next__)
    it = self.pool.imap(sqr, exception_throwing_generator(10, 3), 1)
    for i in range(3):
        self.assertEqual(next(it), i * i)
    self.assertRaises(SayWhenError, it.__next__)
    it = self.pool.imap(sqr, exception_throwing_generator(20, 7), 2)
    for i in range(6):
        self.assertEqual(next(it), i * i)
    self.assertRaises(SayWhenError, it.__next__)
    it = self.pool.imap(sqr, exception_throwing_generator(20, 7), 4)
    for i in range(4):
        self.assertEqual(next(it), i * i)
    self.assertRaises(SayWhenError, it.__next__)