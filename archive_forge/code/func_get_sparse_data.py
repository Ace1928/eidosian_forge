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
def get_sparse_data(pattern):
    basesize = pattern.shape
    assert len(basesize) == 2, basesize
    indices = torch.where(pattern != 0)
    coo_indices = torch.stack(indices)
    crow_indices = torch.zeros(basesize[0] + 1, dtype=torch.int64)
    crow_indices[1:] = torch.cumsum(coo_indices[0].bincount(minlength=basesize[0]), 0)
    col_indices = coo_indices[1]
    strided_values = torch.zeros(basesize, dtype=torch.int64)
    values = torch.arange(1, 1 + len(indices[0]), dtype=torch.int64)
    strided_values[indices] = values
    indices_T = torch.where(pattern.transpose(0, 1) != 0)
    coo_indices_T = torch.stack(indices_T)
    ccol_indices = torch.zeros(basesize[1] + 1, dtype=torch.int64)
    ccol_indices[1:] = torch.cumsum(coo_indices_T[0].bincount(minlength=basesize[1]), 0)
    row_indices = coo_indices_T[1]
    csc_values = strided_values.transpose(0, 1)[indices_T]
    return {torch.sparse_coo: (coo_indices, values), torch.sparse_csr: (crow_indices, col_indices, values), torch.sparse_csc: (ccol_indices, row_indices, csc_values), torch.strided: (strided_values,)}