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
def read_func_attr_all(self):
    attr = binding.CUfunction_attribute
    nregs = self.read_func_attr(attr.CU_FUNC_ATTRIBUTE_NUM_REGS)
    cmem = self.read_func_attr(attr.CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES)
    lmem = self.read_func_attr(attr.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)
    smem = self.read_func_attr(attr.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)
    maxtpb = self.read_func_attr(attr.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    return FuncAttr(regs=nregs, const=cmem, local=lmem, shared=smem, maxthreads=maxtpb)