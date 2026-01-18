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
@unittest.skipUnless(util.abstract_sockets_supported, 'test needs abstract socket support')
def test_abstract_socket(self):
    with self.connection.Listener('\x00something') as listener:
        with self.connection.Client(listener.address) as client:
            with listener.accept() as d:
                client.send(1729)
                self.assertEqual(d.recv(), 1729)
    if self.TYPE == 'processes':
        self.assertRaises(OSError, listener.accept)