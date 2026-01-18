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
@wraps(fn)
def efail_fn(slf, *args, **kwargs):
    if self.device_type is None or self.device_type == slf.device_type:
        try:
            fn(slf, *args, **kwargs)
        except Exception:
            return
        else:
            slf.fail('expected test to fail, but it passed')
    return fn(slf, *args, **kwargs)