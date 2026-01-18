import errno
import fcntl
import multiprocessing
import os
import shutil
import signal
import tempfile
import threading
import time
from fasteners import process_lock as pl
from fasteners import test
def test_interprocess_lock(self):
    lock_file = os.path.join(self.lock_dir, 'lock')
    pid = os.fork()
    if pid:
        start = time.time()
        while not os.path.exists(lock_file):
            if time.time() - start > 5:
                self.fail('Timed out waiting for child to grab lock')
            time.sleep(0)
        lock1 = pl.InterProcessLock('foo')
        lock1.lockfile = open(lock_file, 'w')
        while time.time() - start < 5:
            try:
                lock1.trylock()
                lock1.unlock()
                time.sleep(0)
            except IOError:
                break
        else:
            self.fail('Never caught expected lock exception')
        os.kill(pid, signal.SIGKILL)
    else:
        try:
            lock2 = pl.InterProcessLock('foo')
            lock2.lockfile = open(lock_file, 'w')
            have_lock = False
            while not have_lock:
                try:
                    lock2.trylock()
                    have_lock = True
                except IOError:
                    pass
        finally:
            time.sleep(0.5)
            os._exit(0)