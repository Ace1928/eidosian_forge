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
def random_sparse_matrix(rows, columns, density=0.01, **kwargs):
    """Return rectangular random sparse matrix within given density.

    The density of the result approaches to given density as the size
    of the matrix is increased and a relatively small value of density
    is specified but higher than min(rows, columns)/(rows * columns)
    for non-singular matrices.
    """
    dtype = kwargs.get('dtype', torch.double)
    device = kwargs.get('device', 'cpu')
    singular = kwargs.get('singular', False)
    k = min(rows, columns)
    nonzero_elements = max(min(rows, columns), int(rows * columns * density))
    row_indices = [i % rows for i in range(nonzero_elements)]
    column_indices = [i % columns for i in range(nonzero_elements)]
    random.shuffle(column_indices)
    indices = [row_indices, column_indices]
    values = torch.randn(nonzero_elements, dtype=dtype, device=device)
    values *= torch.tensor([-float(i - j) ** 2 for i, j in zip(*indices)], dtype=dtype, device=device).exp()
    indices_tensor = torch.tensor(indices)
    A = torch.sparse_coo_tensor(indices_tensor, values, (rows, columns), device=device)
    return A.coalesce()