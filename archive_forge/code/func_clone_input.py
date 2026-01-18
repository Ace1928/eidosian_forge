import atexit
import collections
import contextlib
import copy
import cProfile
import dataclasses
import datetime
import dis
import enum
import functools
import gc
import inspect
import itertools
import linecache
import logging
import math
import operator
import os
import pstats
import subprocess
import sys
import textwrap
import threading
import time
import types
import typing
import weakref
from contextlib import contextmanager
from functools import lru_cache, wraps
from pathlib import Path
from typing import (
import importlib
import torch
import torch._functorch.config
import torch.fx.experimental.symbolic_shapes
from torch import fx
from torch._dispatch.python import enable_python_dispatcher
from torch.nn.modules.lazy import LazyModuleMixin
from torch.utils._pytree import tree_map_only
from torch._subclasses import (  # noqa: F401
def clone_input(x, *, dtype=None):
    """copy while preserving strides"""
    if is_fake(x):
        return x

    def torch_clone(x):
        y = torch.clone(x)
        if x.is_leaf:
            y.requires_grad_(x.requires_grad)
        if x.is_leaf and x.grad is not None:
            y.grad = clone_input(x.grad, dtype=dtype)
        if hasattr(x, '_dynamo_dynamic_indices'):
            y._dynamo_dynamic_indices = x._dynamo_dynamic_indices.copy()
        return y
    with torch.no_grad():
        if x.device.type == 'xla':
            return torch_clone(x)
        needed_size = sum(((shape - 1) * stride for shape, stride in zip(x.size(), x.stride())))
        if x.is_quantized:
            result = torch.empty_quantized((needed_size + 32,), x)
        else:
            result = torch.empty(needed_size + 32, dtype=dtype or x.dtype, device=x.device)
        cache_line_offset = (x.data_ptr() - result.data_ptr()) % 32 // x.element_size()
        result.as_strided_(x.size(), x.stride(), cache_line_offset)
        try:
            result.copy_(x.clone())
            if x.is_leaf:
                result.requires_grad_(x.requires_grad)
            if x.is_leaf and x.grad is not None:
                result.grad = clone_input(x.grad, dtype=dtype)
        except RuntimeError:
            return torch_clone(x)
        if hasattr(x, '_dynamo_dynamic_indices'):
            result._dynamo_dynamic_indices = x._dynamo_dynamic_indices.copy()
        return result