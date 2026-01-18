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
@unittest.skipIf(sys.hexversion <= 50988528, 'SemLock subclass')
def test_global_named_resource_spawn(self):
    testfn = os_helper.TESTFN
    self.addCleanup(os_helper.unlink, testfn)
    with open(testfn, 'w', encoding='utf-8') as f:
        f.write(textwrap.dedent("                import multiprocess as mp\n\n                ctx = mp.get_context('spawn')\n\n                global_resource = ctx.Semaphore()\n\n                def submain(): pass\n\n                if __name__ == '__main__':\n                    p = ctx.Process(target=submain)\n                    p.start()\n                    p.join()\n            "))
    rc, out, err = test.support.script_helper.assert_python_ok(testfn, **ENV)
    self.assertEqual(err, b'')