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
def get_devptr_for_active_ctx(ptr):
    """Query the device pointer usable in the current context from an arbitrary
    pointer.
    """
    if ptr != 0:
        if USE_NV_BINDING:
            ptr_attrs = binding.CUpointer_attribute
            attr = ptr_attrs.CU_POINTER_ATTRIBUTE_DEVICE_POINTER
            ptrobj = binding.CUdeviceptr(ptr)
            return driver.cuPointerGetAttribute(attr, ptrobj)
        else:
            devptr = drvapi.cu_device_ptr()
            attr = enums.CU_POINTER_ATTRIBUTE_DEVICE_POINTER
            driver.cuPointerGetAttribute(byref(devptr), attr, ptr)
            return devptr
    elif USE_NV_BINDING:
        return binding.CUdeviceptr()
    else:
        return drvapi.cu_device_ptr()