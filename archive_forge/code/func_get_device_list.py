from __future__ import annotations
import contextlib
import dataclasses
import functools
import logging
import os
import queue
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from ctypes import byref, c_size_t, c_void_p
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from typing import (
import torch
from torch import multiprocessing
from torch._dynamo.testing import rand_strided
from torch._inductor import ir
from torch._inductor.codecache import CUDACodeCache, DLLWrapper, PyCodeCache
from . import config
from .utils import do_bench
from .virtualized import V
def get_device_list(self) -> Sequence[Optional[int]]:
    """
        Gather the list of devices to be used in the pool.
        """
    if not config.autotune_multi_device:
        return [None]
    count = torch.cuda.device_count()
    if CUDA_VISIBLE_DEVICES in os.environ:
        devices = [int(d) for d in os.environ[CUDA_VISIBLE_DEVICES].split(',')]
        assert len(devices) <= count
        return devices
    return list(range(count))