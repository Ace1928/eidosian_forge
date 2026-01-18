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
def random_hermitian_pd_matrix(matrix_size, *batch_dims, dtype, device):
    """
    Returns a batch of random Hermitian positive-definite matrices.
    The shape of the result is batch_dims + (matrix_size, matrix_size)
    The following example creates a tensor of size 2 x 4 x 3 x 3
    >>> # xdoctest: +SKIP("undefined variables")
    >>> matrices = random_hermitian_pd_matrix(3, 2, 4, dtype=dtype, device=device)
    """
    A = torch.randn(*batch_dims + (matrix_size, matrix_size), dtype=dtype, device=device)
    return A @ A.mH + torch.eye(matrix_size, dtype=dtype, device=device)