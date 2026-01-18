import collections
import random
import threading
import time
from concurrent import futures
import fasteners
from fasteners import test
from fasteners import _utils
def test_double_writer(self):
    lock = fasteners.ReaderWriterLock()
    with lock.write_lock():
        self.assertFalse(lock.is_reader())
        self.assertTrue(lock.is_writer())
        with lock.write_lock():
            self.assertTrue(lock.is_writer())
        self.assertTrue(lock.is_writer())
    self.assertFalse(lock.is_reader())
    self.assertFalse(lock.is_writer())