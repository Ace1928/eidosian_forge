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
def generate_values(base, densesize):
    """Generates a tensor of shape densesize with values equal to

              base + i_1 * 10^0 + ... + i_d * 10^{d - 1}

            at indices i_1, ..., i_d (with 0 <= i_j < densesize[j] for any 1 <= j <=
            len(densesize))

            This mapping produces unique values as long as
            densesize[i] < 10 for all i in range(len(densesize)).
            """
    if not densesize:
        return base
    if not isinstance(base, int) and base.ndim > 0:
        return torch.stack([generate_values(b, densesize) for b in base])
    if base == 0:
        return torch.zeros(densesize, dtype=torch.int64)
    r = torch.arange(densesize[0], dtype=torch.int64)
    for i, d in enumerate(densesize[1:]):
        y = torch.arange(d, dtype=torch.int64) * 10 ** (i + 1)
        r = r[..., None] + y[None, ...]
    r.add_(base)
    return r