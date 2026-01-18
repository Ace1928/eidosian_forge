import collections
import random
import threading
import time
from concurrent import futures
import fasteners
from fasteners import test
from fasteners import _utils
def test_double_reader_writer(self):
    lock = fasteners.ReaderWriterLock()
    activated = collections.deque()
    active = threading.Event()

    def double_reader():
        with lock.read_lock():
            active.set()
            while not lock.has_pending_writers:
                time.sleep(0.001)
            with lock.read_lock():
                activated.append(lock.owner)

    def happy_writer():
        with lock.write_lock():
            activated.append(lock.owner)
    reader = _daemon_thread(double_reader)
    reader.start()
    active.wait(WAIT_TIMEOUT)
    self.assertTrue(active.is_set())
    writer = _daemon_thread(happy_writer)
    writer.start()
    reader.join()
    writer.join()
    self.assertEqual(2, len(activated))
    self.assertEqual(['r', 'w'], list(activated))