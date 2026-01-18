import faulthandler
import itertools
import multiprocessing
import os
import random
import re
import subprocess
import sys
import textwrap
import threading
import unittest
import numpy as np
from numba import jit, vectorize, guvectorize, set_num_threads
from numba.tests.support import (temp_directory, override_config, TestCase, tag,
import queue as t_queue
from numba.testing.main import _TIMEOUT as _RUNNER_TIMEOUT
from numba.core import config
def compile_factory(parallel_class, queue_impl):

    def run_compile(fnlist):
        q = queue_impl()
        kws = {'queue': q}
        ths = [parallel_class(target=chooser, args=(fnlist,), kwargs=kws) for i in range(4)]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        if not q.empty():
            errors = []
            while not q.empty():
                errors.append(q.get(False))
            _msg = 'Error(s) occurred in delegated runner:\n%s'
            raise RuntimeError(_msg % '\n'.join([repr(x) for x in errors]))
    return run_compile