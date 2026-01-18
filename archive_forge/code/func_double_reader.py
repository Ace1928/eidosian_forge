import collections
import random
import threading
import time
from concurrent import futures
import fasteners
from fasteners import test
from fasteners import _utils
def double_reader():
    with lock.read_lock():
        active.set()
        while not lock.has_pending_writers:
            time.sleep(0.001)
        with lock.read_lock():
            activated.append(lock.owner)