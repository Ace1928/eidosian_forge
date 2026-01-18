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
def tearDownModule():
    need_sleep = False
    test.support.gc_collect()
    multiprocessing.set_start_method(old_start_method[0], force=True)
    processes = set(multiprocessing.process._dangling) - set(dangling[0])
    if processes:
        need_sleep = True
        test.support.environment_altered = True
        support.print_warning(f'Dangling processes: {processes}')
    processes = None
    threads = set(threading._dangling) - set(dangling[1])
    if threads:
        need_sleep = True
        test.support.environment_altered = True
        support.print_warning(f'Dangling threads: {threads}')
    threads = None
    if need_sleep:
        time.sleep(0.5)
    multiprocessing.util._cleanup_tests()