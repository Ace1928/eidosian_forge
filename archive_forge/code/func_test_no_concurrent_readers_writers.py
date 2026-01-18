import collections
import random
import threading
import time
from concurrent import futures
import fasteners
from fasteners import test
from fasteners import _utils
def test_no_concurrent_readers_writers(self):
    lock = fasteners.ReaderWriterLock()
    watch = _utils.StopWatch(duration=5)
    watch.start()
    dups = collections.deque()
    active = collections.deque()

    def acquire_check(me, reader):
        if reader:
            lock_func = lock.read_lock
        else:
            lock_func = lock.write_lock
        with lock_func():
            if not reader:
                if len(active) >= 1:
                    dups.append(me)
                    dups.extend(active)
            active.append(me)
            try:
                time.sleep(random.random() / 100)
            finally:
                active.remove(me)

    def run():
        me = threading.current_thread()
        while not watch.expired():
            acquire_check(me, random.choice([True, False]))
    threads = []
    for i in range(0, self.THREAD_COUNT):
        t = _daemon_thread(run)
        threads.append(t)
        t.start()
    while threads:
        t = threads.pop()
        t.join()
    self.assertEqual([], list(dups))
    self.assertEqual([], list(active))