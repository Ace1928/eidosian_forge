import collections
import random
import threading
import time
from concurrent import futures
import fasteners
from fasteners import test
from fasteners import _utils
def test_double_reader(self):
    lock = fasteners.ReaderWriterLock()
    with lock.read_lock():
        self.assertTrue(lock.is_reader())
        self.assertFalse(lock.is_writer())
        with lock.read_lock():
            self.assertTrue(lock.is_reader())
        self.assertTrue(lock.is_reader())
    self.assertFalse(lock.is_reader())
    self.assertFalse(lock.is_writer())