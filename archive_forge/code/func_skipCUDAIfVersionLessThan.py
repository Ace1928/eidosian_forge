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
def skipCUDAIfVersionLessThan(versions: Tuple[int, int]=None):

    def dec_fn(fn):

        @wraps(fn)
        def wrap_fn(self, *args, **kwargs):
            version = _get_torch_cuda_version()
            if version == (0, 0):
                return fn(self, *args, **kwargs)
            if version < versions:
                reason = f'test skipped for CUDA versions < {version}'
                raise unittest.SkipTest(reason)
            return fn(self, *args, **kwargs)
        return wrap_fn
    return dec_fn