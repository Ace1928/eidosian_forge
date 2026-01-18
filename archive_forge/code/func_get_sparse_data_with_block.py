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
def get_sparse_data_with_block(pattern, blocksize):
    nonblock_data = get_sparse_data(pattern)
    blockpattern = get_blockpattern(pattern, blocksize)
    block_data = get_sparse_data(blockpattern)
    strided_values = nonblock_data[torch.strided][0]
    block_indices = block_data[torch.sparse_coo][0]
    bsr_values = torch.stack([strided_values[bi * blocksize[0]:(bi + 1) * blocksize[0], bj * blocksize[1]:(bj + 1) * blocksize[1]] for bi, bj in block_indices.transpose(0, 1)])
    bsc_values = bsr_values[block_data[torch.sparse_csc][2] - 1]
    return {torch.sparse_bsr: (*block_data[torch.sparse_csr][:2], bsr_values), torch.sparse_bsc: (*block_data[torch.sparse_csc][:2], bsc_values), **nonblock_data}