import collections
import random
import threading
import time
from concurrent import futures
import fasteners
from fasteners import test
from fasteners import _utils
def chaotic_writer(blow_up):
    with lock.write_lock():
        if blow_up:
            raise RuntimeError('Broken')
        else:
            activated.append(lock.owner)