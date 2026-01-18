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
def random_matrix(rows, columns, *batch_dims, **kwargs):
    """Return rectangular matrix or batches of rectangular matrices.

    Parameters:
      dtype - the data type
      device - the device kind
      singular - when True, the output will be singular
    """
    dtype = kwargs.get('dtype', torch.double)
    device = kwargs.get('device', 'cpu')
    silent = kwargs.get('silent', False)
    singular = kwargs.get('singular', False)
    if silent and (not torch._C.has_lapack):
        return torch.ones(rows, columns, dtype=dtype, device=device)
    A = torch.randn(batch_dims + (rows, columns), dtype=dtype, device=device)
    if A.numel() == 0:
        return A
    u, _, vh = torch.linalg.svd(A, full_matrices=False)
    k = min(rows, columns)
    s = torch.linspace(1 / (k + 1), 1, k, dtype=dtype, device=device)
    if singular:
        s[k - 1] = 0
        if k > 2:
            s[0] = 0
    return u * s.unsqueeze(-2) @ vh