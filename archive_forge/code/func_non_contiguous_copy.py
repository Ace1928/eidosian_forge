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
def non_contiguous_copy(t, dim=-1, offset=0):
    self.assertTrue(t.is_contiguous())
    if dim < 0:
        dim = dim + t.ndim
    assert dim >= 0 and dim < t.ndim
    step = max(2, offset + 1)
    tmp = torch.zeros((*t.shape[:dim], t.shape[dim] * step, *t.shape[dim + 1:]), dtype=t.dtype, device=t.device)
    dim_slices = (*(slice(None),) * dim, slice(offset, None, step))
    r = tmp[dim_slices].copy_(t)
    self.assertFalse(r.is_contiguous())
    self.assertEqual(t, r)
    return r