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
def assertExpected(self, s, subname=None):
    """
        Test that a string matches the recorded contents of a file
        derived from the name of this test and subname.  This file
        is placed in the 'expect' directory in the same directory
        as the test script. You can automatically update the recorded test
        output using --accept.

        If you call this multiple times in a single function, you must
        give a unique subname each time.
        """
    if not isinstance(s, str):
        raise TypeError('assertExpected is strings only')

    def remove_prefix(text, prefix):
        if text.startswith(prefix):
            return text[len(prefix):]
        return text
    module_id = self.__class__.__module__
    munged_id = remove_prefix(self.id(), module_id + '.')
    test_file = os.path.realpath(sys.modules[module_id].__file__)
    expected_file = os.path.join(os.path.dirname(test_file), 'expect', munged_id)
    subname_output = ''
    if subname:
        expected_file += '-' + subname
        subname_output = f' ({subname})'
    expected_file += '.expect'
    expected = None

    def accept_output(update_type):
        print(f'Accepting {update_type} for {munged_id}{subname_output}:\n\n{s}')
        with open(expected_file, 'w') as f:
            s_tag = re.sub('(producer_version): "[0-9.]*"', '\\1: "CURRENT_VERSION"', s)
            f.write(s_tag)
    try:
        with open(expected_file) as f:
            expected = f.read()
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise
        elif expecttest.ACCEPT:
            return accept_output('output')
        else:
            raise RuntimeError(f'I got this output for {munged_id}{subname_output}:\n\n{s}\n\nNo expect file exists; to accept the current output, run:\npython {__main__.__file__} {munged_id} --accept') from None
    if IS_WINDOWS:
        expected = re.sub('CppOp\\[(.+?)\\]', 'CppOp[]', expected)
        s = re.sub('CppOp\\[(.+?)\\]', 'CppOp[]', s)
    expected = expected.replace('producer_version: "CURRENT_VERSION"', f'producer_version: "{torch.onnx.producer_version}"')
    if expecttest.ACCEPT:
        if expected != s:
            return accept_output('updated output')
    elif hasattr(self, 'assertMultiLineEqual'):
        self.assertMultiLineEqual(expected, s)
    else:
        self.assertEqual(s, expected)