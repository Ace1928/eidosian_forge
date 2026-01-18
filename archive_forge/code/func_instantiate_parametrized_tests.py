import argparse
import contextlib
import copy
import ctypes
import errno
import functools
import gc
import inspect
import io
import json
import logging
import math
import operator
import os
import platform
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import warnings
from collections.abc import Mapping, Sequence
from contextlib import closing, contextmanager
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from functools import partial, wraps
from itertools import product, chain
from pathlib import Path
from statistics import mean
from typing import (
from unittest.mock import MagicMock
import expecttest
import numpy as np
import __main__  # type: ignore[import]
import torch
import torch.backends.cudnn
import torch.backends.mkl
import torch.backends.mps
import torch.backends.xnnpack
import torch.cuda
from torch import Tensor
from torch._C import ScriptDict, ScriptList  # type: ignore[attr-defined]
from torch._utils_internal import get_writable_path
from torch.nn import (
from torch.onnx import (
from torch.testing import make_tensor
from torch.testing._comparison import (
from torch.testing._comparison import not_close_error_metas
from torch.testing._internal.common_dtype import get_all_dtypes
import torch.utils._pytree as pytree
from .composite_compliance import no_dispatch
def instantiate_parametrized_tests(generic_cls):
    """
    Instantiates tests that have been decorated with a parametrize_fn. This is generally performed by a
    decorator subclass of _TestParametrizer. The generic test will be replaced on the test class by
    parametrized tests with specialized names. This should be used instead of
    instantiate_device_type_tests() if the test class contains device-agnostic tests.

    You can also use it as a class decorator. E.g.

    ```
    @instantiate_parametrized_tests
    class TestFoo(TestCase):
        ...
    ```

    Args:
        generic_cls (class): Generic test class object containing tests (e.g. TestFoo)
    """
    for attr_name in tuple(dir(generic_cls)):
        class_attr = getattr(generic_cls, attr_name)
        if not hasattr(class_attr, 'parametrize_fn'):
            continue
        delattr(generic_cls, attr_name)

        def instantiate_test_helper(cls, name, test, param_kwargs):

            @wraps(test)
            def instantiated_test(self, param_kwargs=param_kwargs):
                test(self, **param_kwargs)
            assert not hasattr(generic_cls, name), f'Redefinition of test {name}'
            setattr(generic_cls, name, instantiated_test)
        for test, test_suffix, param_kwargs, decorator_fn in class_attr.parametrize_fn(class_attr, generic_cls=generic_cls, device_cls=None):
            full_name = f'{test.__name__}_{test_suffix}'
            for decorator in decorator_fn(param_kwargs):
                test = decorator(test)
            instantiate_test_helper(cls=generic_cls, name=full_name, test=test, param_kwargs=param_kwargs)
    return generic_cls