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
@pl.interprocess_locked(os.path.join(self.lock_dir, 'test-lock-2'))
def inner_lock():
    return sentinel