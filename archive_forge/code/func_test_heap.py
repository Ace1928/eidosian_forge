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
def test_heap(self):
    iterations = 5000
    maxblocks = 50
    blocks = []
    heap = multiprocessing.heap.BufferWrapper._heap
    heap._DISCARD_FREE_SPACE_LARGER_THAN = 0
    for i in range(iterations):
        size = int(random.lognormvariate(0, 1) * 1000)
        b = multiprocessing.heap.BufferWrapper(size)
        blocks.append(b)
        if len(blocks) > maxblocks:
            i = random.randrange(maxblocks)
            del blocks[i]
        del b
    with heap._lock:
        all = []
        free = 0
        occupied = 0
        for L in list(heap._len_to_seq.values()):
            for arena, start, stop in L:
                all.append((heap._arenas.index(arena), start, stop, stop - start, 'free'))
                free += stop - start
        for arena, arena_blocks in heap._allocated_blocks.items():
            for start, stop in arena_blocks:
                all.append((heap._arenas.index(arena), start, stop, stop - start, 'occupied'))
                occupied += stop - start
        self.assertEqual(free + occupied, sum((arena.size for arena in heap._arenas)))
        all.sort()
        for i in range(len(all) - 1):
            arena, start, stop = all[i][:3]
            narena, nstart, nstop = all[i + 1][:3]
            if arena != narena:
                self.assertEqual(stop, heap._arenas[arena].size)
                self.assertEqual(nstart, 0)
            else:
                self.assertEqual(stop, nstart)
    random.shuffle(blocks)
    while blocks:
        blocks.pop()
    self.assertEqual(heap._n_frees, heap._n_mallocs)
    self.assertEqual(len(heap._pending_free_blocks), 0)
    self.assertEqual(len(heap._arenas), 0)
    self.assertEqual(len(heap._allocated_blocks), 0, heap._allocated_blocks)
    self.assertEqual(len(heap._len_to_seq), 0)