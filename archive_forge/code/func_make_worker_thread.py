import itertools
import threading
import time
import numpy as np
from numpy.testing import assert_equal
import pytest
import scipy.interpolate
def make_worker_thread(self, target, args):
    log = self.log

    class WorkerThread(threading.Thread):

        def run(self):
            log('interpolation started')
            target(*args)
            log('interpolation complete')
    return WorkerThread()