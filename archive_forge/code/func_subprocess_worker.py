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
@contextmanager
def subprocess_worker(protocol=None):
    worker = _Worker(protocol=protocol)
    yield worker
    worker.close()