import sys
import os
import os.path as op
import tempfile
from subprocess import Popen, check_output, PIPE, STDOUT, CalledProcessError
from srsly.cloudpickle.compat import pickle
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
import psutil
from srsly.cloudpickle import dumps
from subprocess import TimeoutExpired
def make_local_function():

    def g(x):
        assert TEST_GLOBALS == 'a test value'
        return sum(range(10))
    return g