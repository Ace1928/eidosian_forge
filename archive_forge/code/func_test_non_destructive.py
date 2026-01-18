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
def test_non_destructive(self):
    lock_file = os.path.join(self.lock_dir, 'not-destroyed')
    with open(lock_file, 'w') as f:
        f.write('test')
    with pl.InterProcessLock(lock_file):
        with open(lock_file) as f:
            self.assertEqual(f.read(), 'test')