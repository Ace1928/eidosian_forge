import traceback
import threading
import multiprocessing
import numpy as np
from numba import cuda
from numba.cuda.testing import (skip_on_cudasim, skip_under_cuda_memcheck,
import unittest
def spawn_process_entry(q):
    try:
        check_concurrent_compiling()
    except:
        msg = traceback.format_exc()
        q.put('\n'.join(['', '=' * 80, msg]))
    else:
        q.put(None)