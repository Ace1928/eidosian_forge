import multiprocessing as mp
import itertools
import traceback
import pickle
import numpy as np
from numba import cuda
from numba.cuda.cudadrv import driver
from numba.cuda.testing import (skip_on_arm, skip_on_cudasim,
from numba.tests.support import linux_only, windows_only
import unittest
def staged_ipc_array_test(ipcarr, device_num, result_queue):
    try:
        with cuda.gpus[device_num]:
            with ipcarr as darr:
                arr = darr.copy_to_host()
                try:
                    with ipcarr:
                        pass
                except ValueError as e:
                    if str(e) != 'IpcHandle is already opened':
                        raise AssertionError('invalid exception message')
                else:
                    raise AssertionError('did not raise on reopen')
    except:
        succ = False
        out = traceback.format_exc()
    else:
        succ = True
        out = arr
    result_queue.put((succ, out))