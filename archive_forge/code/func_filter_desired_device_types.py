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
def filter_desired_device_types(device_type_test_bases, except_for=None, only_for=None):
    intersect = set(except_for if except_for else []) & set(only_for if only_for else [])
    assert not intersect, f'device ({intersect}) appeared in both except_for and only_for'
    if except_for:
        device_type_test_bases = filter(lambda x: x.device_type not in except_for, device_type_test_bases)
    if only_for:
        device_type_test_bases = filter(lambda x: x.device_type in only_for, device_type_test_bases)
    return list(device_type_test_bases)