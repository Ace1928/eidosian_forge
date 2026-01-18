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
import functools
import signal
import array
import socket
import random
import logging
import subprocess
import struct
import operator
import pathlib
import pickle #XXX: use dill?
import weakref
import warnings
import test.support
import test.support.script_helper
from test import support
from test.support import hashlib_helper
from test.support import import_helper
from test.support import os_helper
from test.support import script_helper
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
from multiprocess.connection import wait, AuthenticationError
from multiprocess import util
from multiprocess.managers import BaseManager, BaseProxy, RemoteError
def only_run_in_spawn_testsuite(reason):
    """Returns a decorator: raises SkipTest when SM != spawn at test time.

    This can be useful to save overall Python test suite execution time.
    "spawn" is the universal mode available on all platforms so this limits the
    decorated test to only execute within test_multiprocessing_spawn.

    This would not be necessary if we refactored our test suite to split things
    into other test files when they are not start method specific to be rerun
    under all start methods.
    """

    def decorator(test_item):

        @functools.wraps(test_item)
        def spawn_check_wrapper(*args, **kwargs):
            if (start_method := multiprocessing.get_start_method()) != 'spawn':
                raise unittest.SkipTest(f"start_method={start_method!r}, not 'spawn'; {reason}")
            return test_item(*args, **kwargs)
        return spawn_check_wrapper
    return decorator