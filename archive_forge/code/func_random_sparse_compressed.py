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
def random_sparse_compressed(n_compressed_dims, n_plain_dims, nnz):
    compressed_indices = self._make_crow_indices(n_compressed_dims, n_plain_dims, nnz, device=device, dtype=index_dtype)
    plain_indices = torch.zeros(nnz, dtype=index_dtype, device=device)
    for i in range(n_compressed_dims):
        count = compressed_indices[i + 1] - compressed_indices[i]
        plain_indices[compressed_indices[i]:compressed_indices[i + 1]], _ = torch.sort(torch.randperm(n_plain_dims, dtype=index_dtype, device=device)[:count])
    low = -1 if dtype != torch.uint8 else 0
    high = 1 if dtype != torch.uint8 else 2
    values = make_tensor((nnz,) + blocksize + dense_size, device=device, dtype=dtype, low=low, high=high)
    return (values, compressed_indices, plain_indices)