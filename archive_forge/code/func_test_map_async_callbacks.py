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
def test_map_async_callbacks(self):
    call_args = self.manager.list() if self.TYPE == 'manager' else []
    self.pool.map_async(int, ['1'], callback=call_args.append, error_callback=call_args.append).wait()
    self.assertEqual(1, len(call_args))
    self.assertEqual([1], call_args[0])
    self.pool.map_async(int, ['a'], callback=call_args.append, error_callback=call_args.append).wait()
    self.assertEqual(2, len(call_args))
    self.assertIsInstance(call_args[1], ValueError)