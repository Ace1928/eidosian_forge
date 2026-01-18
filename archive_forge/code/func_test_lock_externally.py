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
def test_lock_externally(self):
    self._do_test_lock_externally(self.lock_dir)