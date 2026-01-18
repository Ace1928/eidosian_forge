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
@skip_no_omp
@skip_no_tbb
def test_invalid_env_var(self):
    env_var = 'tbb omp workqueue notvalidhere'
    with self.assertRaises(AssertionError) as raises:
        self.each_env_var(env_var)
    for msg in ('THREADING_LAYER_PRIORITY invalid:', 'It must be a permutation of'):
        self.assertIn(f'{msg}', str(raises.exception))