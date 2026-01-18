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
def run_subprocess(args):
    seed, env, sub_label, extra_env_vars = args
    core_str = None
    try:
        core_str, result_file, num_threads = _WORKER_POOL.get()
        with open(result_file, 'wb'):
            pass
        env_vars: Dict[str, str] = {'PATH': os.getenv('PATH') or '', 'PYTHONPATH': os.getenv('PYTHONPATH') or '', 'OMP_NUM_THREADS': str(num_threads), 'MKL_NUM_THREADS': str(num_threads), 'NUMEXPR_NUM_THREADS': str(num_threads)}
        env_vars.update(extra_env_vars or {})
        subprocess.run(f"source activate {env} && taskset --cpu-list {core_str} python {os.path.abspath(__file__)} --DETAIL-in-subprocess --DETAIL-seed {seed} --DETAIL-num-threads {num_threads} --DETAIL-sub-label '{sub_label}' --DETAIL-result-file {result_file} --DETAIL-env {env}", env=env_vars, stdout=subprocess.PIPE, shell=True)
        with open(result_file, 'rb') as f:
            result_bytes = f.read()
        with _RESULT_FILE_LOCK, open(RESULT_FILE, 'ab') as f:
            f.write(result_bytes)
    except KeyboardInterrupt:
        pass
    finally:
        if core_str is not None:
            _WORKER_POOL.put((core_str, result_file, num_threads))