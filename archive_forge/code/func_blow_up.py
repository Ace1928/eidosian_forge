import collections
import random
import threading
import time
from concurrent import futures
import fasteners
from fasteners import test
from fasteners import _utils
def blow_up():
    with lock.read_lock():
        self.assertEqual(lock.READER, lock.owner)
        raise RuntimeError('Broken')