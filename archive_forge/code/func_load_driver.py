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
def load_driver(dlloader, candidates):
    path_not_exist = []
    driver_load_error = []
    for path in candidates:
        try:
            dll = dlloader(path)
        except OSError as e:
            path_not_exist.append(not os.path.isfile(path))
            driver_load_error.append(e)
        else:
            return (dll, path)
    if all(path_not_exist):
        _raise_driver_not_found()
    else:
        errmsg = '\n'.join((str(e) for e in driver_load_error))
        _raise_driver_error(errmsg)