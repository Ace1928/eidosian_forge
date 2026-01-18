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
def open_ipc_handle(self, handle, size):
    flags = 1
    if USE_NV_BINDING:
        dptr = driver.cuIpcOpenMemHandle(handle, flags)
    else:
        dptr = drvapi.cu_device_ptr()
        driver.cuIpcOpenMemHandle(byref(dptr), handle, flags)
    return MemoryPointer(context=weakref.proxy(self), pointer=dptr, size=size)