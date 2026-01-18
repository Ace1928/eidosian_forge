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
def onlyCUDAAndPRIVATEUSE1(fn):

    @wraps(fn)
    def only_fn(self, *args, **kwargs):
        if self.device_type not in ('cuda', torch._C._get_privateuse1_backend_name()):
            reason = f"onlyCUDAAndPRIVATEUSE1: doesn't run on {self.device_type}"
            raise unittest.SkipTest(reason)
        return fn(self, *args, **kwargs)
    return only_fn