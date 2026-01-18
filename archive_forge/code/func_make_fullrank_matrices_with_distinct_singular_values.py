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
def make_fullrank_matrices_with_distinct_singular_values(*shape, device, dtype, requires_grad=False):
    with torch.no_grad():
        t = make_tensor(shape, device=device, dtype=dtype)
        u, _, vh = torch.linalg.svd(t, full_matrices=False)
        real_dtype = t.real.dtype if t.dtype.is_complex else t.dtype
        k = min(shape[-1], shape[-2])
        s = torch.arange(2, k + 2, dtype=real_dtype, device=device)
        s[1::2] *= -1.0
        s.reciprocal_().add_(1.0)
        x = u * s.to(u.dtype) @ vh
    x.requires_grad_(requires_grad)
    return x