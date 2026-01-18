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
def clear_worker_pool():
    while not _WORKER_POOL.empty():
        _, result_file, _ = _WORKER_POOL.get_nowait()
        os.remove(result_file)
    if os.path.exists(SCRATCH_DIR):
        shutil.rmtree(SCRATCH_DIR)