from __future__ import annotations
import collections
import contextlib
import enum
import functools
import getpass
import inspect
import itertools
import logging
import math
import operator
import os
import platform
import re
import shutil
import sys
import tempfile
import textwrap
import time
import unittest
from io import StringIO
from typing import (
from unittest import mock
import sympy
from typing_extensions import Concatenate, ParamSpec
import torch
from torch._dynamo.device_interface import get_interface_for_device
from torch.autograd import DeviceType
from torch.autograd.profiler_util import EventList
from torch.utils._sympy.functions import CeilDiv, CleanDiv, FloorDiv, ModularIndexing
from . import config
@functools.lru_cache(None)
def get_device_tflops(dtype):
    from triton.testing import get_max_simd_tflops, get_max_tensorcore_tflops
    assert dtype in (torch.float16, torch.bfloat16, torch.float32)
    if torch.version.hip:
        if dtype in (torch.float16, torch.bfloat16):
            return get_max_tensorcore_tflops(dtype)
        if torch.backends.cuda.matmul.allow_tf32:
            return get_max_tensorcore_tflops(torch.float32)
        else:
            return get_max_simd_tflops(torch.float32)
    from triton.testing import nvsmi
    cur_sm_clock = nvsmi(['clocks.current.sm'])[0]
    if dtype in (torch.float16, torch.bfloat16):
        return get_max_tensorcore_tflops(dtype, cur_sm_clock)
    if torch.backends.cuda.matmul.allow_tf32:
        return get_max_tensorcore_tflops(torch.float32, cur_sm_clock)
    else:
        return get_max_simd_tflops(torch.float32, cur_sm_clock)