import sys
import os
import ctypes
import weakref
import functools
import warnings
import logging
import threading
import asyncio
import pathlib
from itertools import product
from abc import ABCMeta, abstractmethod
from ctypes import (c_int, byref, c_size_t, c_char, c_char_p, addressof,
import contextlib
import importlib
import numpy as np
from collections import namedtuple, deque
from numba import mviewbuf
from numba.core import utils, serialize, config
from .error import CudaSupportError, CudaDriverError
from .drvapi import API_PROTOTYPES
from .drvapi import cu_occupancy_b2d_size, cu_stream_callback_pyobj, cu_uuid
from numba.cuda.cudadrv import enums, drvapi, nvrtc, _extras
def locate_driver_and_loader():
    envpath = config.CUDA_DRIVER
    if envpath == '0':
        _raise_driver_not_found()
    if sys.platform == 'win32':
        dlloader = ctypes.WinDLL
        dldir = ['\\windows\\system32']
        dlnames = ['nvcuda.dll']
    elif sys.platform == 'darwin':
        dlloader = ctypes.CDLL
        dldir = ['/usr/local/cuda/lib']
        dlnames = ['libcuda.dylib']
    else:
        dlloader = ctypes.CDLL
        dldir = ['/usr/lib', '/usr/lib64']
        dlnames = ['libcuda.so', 'libcuda.so.1']
    if envpath:
        try:
            envpath = os.path.abspath(envpath)
        except ValueError:
            raise ValueError('NUMBA_CUDA_DRIVER %s is not a valid path' % envpath)
        if not os.path.isfile(envpath):
            raise ValueError('NUMBA_CUDA_DRIVER %s is not a valid file path.  Note it must be a filepath of the .so/.dll/.dylib or the driver' % envpath)
        candidates = [envpath]
    else:
        candidates = dlnames + [os.path.join(x, y) for x, y in product(dldir, dlnames)]
    return (dlloader, candidates)