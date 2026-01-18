import builtins
import collections
import inspect
import itertools
import math
import operator
import warnings
from collections.abc import Iterable
from enum import Enum
from functools import partial, reduce, singledispatch, wraps
from typing import Any, Callable, Dict, List, Optional, overload, Sequence, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
from torch import sym_float, sym_int
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._decomp import register_decomposition
import torch._refs._conversions
import torch._refs.fft
import torch._refs.linalg
import torch._refs.nn.functional
import torch._refs.special
def cat_compute_output_memory_format(inputs):
    format = None
    for t in inputs:
        f = utils.suggest_memory_format(t)
        if f == torch.contiguous_format:
            return f
        if format is not None and format != f:
            return torch.contiguous_format
        format = f
    assert format is not None
    return format