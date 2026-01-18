import copy
import gc
import inspect
import runpy
import sys
import threading
from collections import namedtuple
from enum import Enum
from functools import wraps, partial
from typing import List, Any, ClassVar, Optional, Sequence, Tuple, Union, Dict, Set
import unittest
import os
import torch
from torch.testing._internal.common_utils import TestCase, TEST_WITH_ROCM, TEST_MKL, \
from torch.testing._internal.common_cuda import _get_torch_cuda_version, \
from torch.testing._internal.common_dtype import get_all_dtypes
@classmethod
def get_all_devices(cls):
    primary_device_idx = int(cls.get_primary_device().split(':')[1])
    num_devices = cls.device_mod.device_count()
    prim_device = cls.get_primary_device()
    device_str = f'{cls.device_type}:{{0}}'
    non_primary_devices = [device_str.format(idx) for idx in range(num_devices) if idx != primary_device_idx]
    return [prim_device] + non_primary_devices