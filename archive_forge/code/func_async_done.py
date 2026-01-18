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
def async_done(self) -> asyncio.futures.Future:
    """
        Return an awaitable that resolves once all preceding stream operations
        are complete. The result of the awaitable is the current stream.
        """
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def resolver(future, status):
        if future.done():
            return
        elif status == 0:
            future.set_result(self)
        else:
            future.set_exception(Exception(f'Stream error {status}'))

    def callback(stream, status, future):
        loop.call_soon_threadsafe(resolver, future, status)
    self.add_callback(callback, future)
    return future