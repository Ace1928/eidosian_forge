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
def install_tests_in_module_dict(remote_globs, start_method):
    __module__ = remote_globs['__name__']
    local_globs = globals()
    ALL_TYPES = {'processes', 'threads', 'manager'}
    for name, base in local_globs.items():
        if not isinstance(base, type):
            continue
        if issubclass(base, BaseTestCase):
            if base is BaseTestCase:
                continue
            assert set(base.ALLOWED_TYPES) <= ALL_TYPES, base.ALLOWED_TYPES
            for type_ in base.ALLOWED_TYPES:
                newname = 'With' + type_.capitalize() + name[1:]
                Mixin = local_globs[type_.capitalize() + 'Mixin']

                class Temp(base, Mixin, unittest.TestCase):
                    pass
                if type_ == 'manager':
                    Temp = hashlib_helper.requires_hashdigest('md5')(Temp)
                Temp.__name__ = Temp.__qualname__ = newname
                Temp.__module__ = __module__
                remote_globs[newname] = Temp
        elif issubclass(base, unittest.TestCase):

            class Temp(base, object):
                pass
            Temp.__name__ = Temp.__qualname__ = name
            Temp.__module__ = __module__
            remote_globs[name] = Temp
    dangling = [None, None]
    old_start_method = [None]

    def setUpModule():
        multiprocessing.set_forkserver_preload(PRELOAD)
        multiprocessing.process._cleanup()
        dangling[0] = multiprocessing.process._dangling.copy()
        dangling[1] = threading._dangling.copy()
        old_start_method[0] = multiprocessing.get_start_method(allow_none=True)
        try:
            multiprocessing.set_start_method(start_method, force=True)
        except ValueError:
            raise unittest.SkipTest(start_method + ' start method not supported')
        if sys.platform.startswith('linux'):
            try:
                lock = multiprocessing.RLock()
            except OSError:
                raise unittest.SkipTest('OSError raises on RLock creation, see issue 3111!')
        check_enough_semaphores()
        util.get_temp_dir()
        multiprocessing.get_logger().setLevel(LOG_LEVEL)

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
    remote_globs['setUpModule'] = setUpModule
    remote_globs['tearDownModule'] = tearDownModule