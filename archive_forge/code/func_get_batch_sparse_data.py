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
def get_batch_sparse_data(pattern, blocksize):
    size = pattern.shape
    if len(size) <= 2:
        return get_sparse_data_with_block(pattern, blocksize)
    batch_data = {}
    for i, item in enumerate(pattern):
        for layout, d in get_batch_sparse_data(item, blocksize).items():
            target = batch_data.get(layout)
            if layout is torch.sparse_coo:
                ext_coo_indices1 = torch.cat((torch.full((1, len(d[1])), i, dtype=torch.int64), d[0]))
                if target is None:
                    target = batch_data[layout] = (ext_coo_indices1, d[1])
                else:
                    target[0].set_(torch.cat((target[0], ext_coo_indices1), 1))
                    target[1].set_(torch.cat((target[1], d[1])))
            elif target is None:
                target = batch_data[layout] = tuple((d[j].unsqueeze(0) for j in range(len(d))))
            else:
                for j in range(len(d)):
                    target[j].set_(torch.cat((target[j], d[j].unsqueeze(0))))
    return batch_data