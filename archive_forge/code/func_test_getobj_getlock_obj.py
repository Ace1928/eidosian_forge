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
@unittest.skipIf(c_int is None, 'requires _ctypes')
def test_getobj_getlock_obj(self):
    arr1 = self.Array('i', list(range(10)))
    lock1 = arr1.get_lock()
    obj1 = arr1.get_obj()
    arr2 = self.Array('i', list(range(10)), lock=None)
    lock2 = arr2.get_lock()
    obj2 = arr2.get_obj()
    lock = self.Lock()
    arr3 = self.Array('i', list(range(10)), lock=lock)
    lock3 = arr3.get_lock()
    obj3 = arr3.get_obj()
    self.assertEqual(lock, lock3)
    arr4 = self.Array('i', range(10), lock=False)
    self.assertFalse(hasattr(arr4, 'get_lock'))
    self.assertFalse(hasattr(arr4, 'get_obj'))
    self.assertRaises(AttributeError, self.Array, 'i', range(10), lock='notalock')
    arr5 = self.RawArray('i', range(10))
    self.assertFalse(hasattr(arr5, 'get_lock'))
    self.assertFalse(hasattr(arr5, 'get_obj'))