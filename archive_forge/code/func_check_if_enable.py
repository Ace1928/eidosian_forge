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
def check_if_enable(test: unittest.TestCase):
    classname = str(test.__class__).split("'")[1].split('.')[-1]
    sanitized_testname = remove_device_and_dtype_suffixes(test._testMethodName)

    def matches_test(target: str):
        target_test_parts = target.split()
        if len(target_test_parts) < 2:
            return False
        target_testname = target_test_parts[0]
        target_classname = target_test_parts[1][1:-1].split('.')[-1]
        return classname.startswith(target_classname) and target_testname in (test._testMethodName, sanitized_testname)
    if any((matches_test(x) for x in slow_tests_dict.keys())):
        getattr(test, test._testMethodName).__dict__['slow_test'] = True
        if not TEST_WITH_SLOW:
            raise unittest.SkipTest('test is slow; run with PYTORCH_TEST_WITH_SLOW to enable test')
    if not IS_SANDCASTLE:
        should_skip = False
        skip_msg = ''
        for disabled_test, (issue_url, platforms) in disabled_tests_dict.items():
            if matches_test(disabled_test):
                platform_to_conditional: Dict = {'mac': IS_MACOS, 'macos': IS_MACOS, 'win': IS_WINDOWS, 'windows': IS_WINDOWS, 'linux': IS_LINUX, 'rocm': TEST_WITH_ROCM, 'asan': TEST_WITH_ASAN, 'dynamo': TEST_WITH_TORCHDYNAMO, 'inductor': TEST_WITH_TORCHINDUCTOR, 'slow': TEST_WITH_SLOW}
                invalid_platforms = list(filter(lambda p: p not in platform_to_conditional, platforms))
                if len(invalid_platforms) > 0:
                    invalid_plats_str = ', '.join(invalid_platforms)
                    valid_plats = ', '.join(platform_to_conditional.keys())
                    print(f'Test {disabled_test} is disabled for some unrecognized ', f'platforms: [{invalid_plats_str}]. Please edit issue {issue_url} to fix the platforms ', 'assigned to this flaky test, changing "Platforms: ..." to a comma separated ', f'subset of the following (or leave it blank to match all platforms): {valid_plats}')
                    platforms = list(filter(lambda p: p in platform_to_conditional, platforms))
                if platforms == [] or any((platform_to_conditional[platform] for platform in platforms)):
                    should_skip = True
                    skip_msg = f"Test is disabled because an issue exists disabling it: {issue_url} for {('all' if platforms == [] else '')}platform(s) {', '.join(platforms)}. If you're seeing this on your local machine and would like to enable this test, please make sure CI is not set and you are not using the flag --import-disabled-tests."
                    break
        if should_skip and (not RERUN_DISABLED_TESTS):
            raise unittest.SkipTest(skip_msg)
        if not should_skip and RERUN_DISABLED_TESTS:
            skip_msg = 'Test is enabled but --rerun-disabled-tests verification mode is set, so only disabled tests are run'
            raise unittest.SkipTest(skip_msg)
    if TEST_SKIP_FAST:
        if hasattr(test, test._testMethodName) and (not getattr(test, test._testMethodName).__dict__.get('slow_test', False)):
            raise unittest.SkipTest('test is fast; we disabled it with PYTORCH_TEST_SKIP_FAST')