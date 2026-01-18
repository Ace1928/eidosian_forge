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
def attempt_acquire(count):
    children = []
    for i in range(count):
        child = multiprocessing.Process(target=try_lock)
        child.start()
        children.append(child)
    exit_codes = []
    for child in children:
        child.join()
        exit_codes.append(child.exitcode)
    return sum(exit_codes)