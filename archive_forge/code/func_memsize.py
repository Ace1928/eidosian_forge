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
def memsize(self):
    workers_pids = [p.pid if hasattr(p, 'pid') else p for p in list(self.pool._processes)]
    num_workers = len(workers_pids)
    if num_workers == 0:
        return 0
    elif num_workers > 1:
        raise RuntimeError('Unexpected number of workers: %d' % num_workers)
    return psutil.Process(workers_pids[0]).memory_info().rss