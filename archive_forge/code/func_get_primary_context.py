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
def get_primary_context(self):
    """
        Returns the primary context for the device.
        Note: it is not pushed to the CPU thread.
        """
    if self.primary_context is not None:
        return self.primary_context
    met_requirement_for_device(self)
    if USE_NV_BINDING:
        hctx = driver.cuDevicePrimaryCtxRetain(self.id)
    else:
        hctx = drvapi.cu_context()
        driver.cuDevicePrimaryCtxRetain(byref(hctx), self.id)
    ctx = Context(weakref.proxy(self), hctx)
    self.primary_context = ctx
    return ctx