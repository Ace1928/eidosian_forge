import argparse
import datetime
import itertools as it
import multiprocessing
import multiprocessing.dummy
import os
import queue
import pickle
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Tuple, Dict
from . import blas_compare_setup
def fill_core_pool(n: int):
    clear_worker_pool()
    os.makedirs(SCRATCH_DIR)
    cpu_count = multiprocessing.cpu_count() - 2
    step = max(n, 2)
    for i in range(0, cpu_count, step):
        core_str = f'{i}' if n == 1 else f'{i},{i + n - 1}'
        _, result_file = tempfile.mkstemp(suffix='.pkl', prefix=SCRATCH_DIR)
        _WORKER_POOL.put((core_str, result_file, n))