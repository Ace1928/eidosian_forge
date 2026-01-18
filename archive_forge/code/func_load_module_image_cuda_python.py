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
def load_module_image_cuda_python(context, image):
    """
    image must be a pointer
    """
    logsz = config.CUDA_LOG_SIZE
    jitinfo = bytearray(logsz)
    jiterrors = bytearray(logsz)
    jit_option = binding.CUjit_option
    options = {jit_option.CU_JIT_INFO_LOG_BUFFER: jitinfo, jit_option.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: logsz, jit_option.CU_JIT_ERROR_LOG_BUFFER: jiterrors, jit_option.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: logsz, jit_option.CU_JIT_LOG_VERBOSE: config.CUDA_VERBOSE_JIT_LOG}
    option_keys = [k for k in options.keys()]
    option_vals = [v for v in options.values()]
    try:
        handle = driver.cuModuleLoadDataEx(image, len(options), option_keys, option_vals)
    except CudaAPIError as e:
        err_string = jiterrors.decode('utf-8')
        msg = 'cuModuleLoadDataEx error:\n%s' % err_string
        raise CudaAPIError(e.code, msg)
    info_log = jitinfo.decode('utf-8')
    return CudaPythonModule(weakref.proxy(context), handle, info_log, _module_finalizer(context, handle))