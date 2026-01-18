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
def get_device_type_test_bases():
    test_bases: List[Any] = list()
    if IS_SANDCASTLE or IS_FBCODE:
        if IS_REMOTE_GPU:
            if not TEST_WITH_ASAN and (not TEST_WITH_TSAN) and (not TEST_WITH_UBSAN):
                test_bases.append(CUDATestBase)
        else:
            test_bases.append(CPUTestBase)
    else:
        test_bases.append(CPUTestBase)
        if torch.cuda.is_available():
            test_bases.append(CUDATestBase)
        device_type = torch._C._get_privateuse1_backend_name()
        device_mod = getattr(torch, device_type, None)
        if hasattr(device_mod, 'is_available') and device_mod.is_available():
            test_bases.append(PrivateUse1TestBase)
    return test_bases