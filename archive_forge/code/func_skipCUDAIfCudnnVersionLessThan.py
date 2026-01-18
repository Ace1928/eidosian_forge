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
def skipCUDAIfCudnnVersionLessThan(version=0):

    def dec_fn(fn):

        @wraps(fn)
        def wrap_fn(self, *args, **kwargs):
            if self.device_type == 'cuda':
                if self.no_cudnn:
                    reason = 'cuDNN not available'
                    raise unittest.SkipTest(reason)
                if self.cudnn_version is None or self.cudnn_version < version:
                    reason = f'cuDNN version {self.cudnn_version} is available but {version} required'
                    raise unittest.SkipTest(reason)
            return fn(self, *args, **kwargs)
        return wrap_fn
    return dec_fn