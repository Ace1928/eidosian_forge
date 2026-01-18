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
def load_module_image_ctypes(context, image):
    logsz = config.CUDA_LOG_SIZE
    jitinfo = (c_char * logsz)()
    jiterrors = (c_char * logsz)()
    options = {enums.CU_JIT_INFO_LOG_BUFFER: addressof(jitinfo), enums.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: c_void_p(logsz), enums.CU_JIT_ERROR_LOG_BUFFER: addressof(jiterrors), enums.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: c_void_p(logsz), enums.CU_JIT_LOG_VERBOSE: c_void_p(config.CUDA_VERBOSE_JIT_LOG)}
    option_keys = (drvapi.cu_jit_option * len(options))(*options.keys())
    option_vals = (c_void_p * len(options))(*options.values())
    handle = drvapi.cu_module()
    try:
        driver.cuModuleLoadDataEx(byref(handle), image, len(options), option_keys, option_vals)
    except CudaAPIError as e:
        msg = 'cuModuleLoadDataEx error:\n%s' % jiterrors.value.decode('utf8')
        raise CudaAPIError(e.code, msg)
    info_log = jitinfo.value
    return CtypesModule(weakref.proxy(context), handle, info_log, _module_finalizer(context, handle))