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
def defake(x):
    if not isinstance(x, FakeTensor):
        return x
    size: 'torch._prims_common.ShapeType'
    stride: 'torch._prims_common.StrideType'
    if x._has_symbolic_sizes_strides:
        size = [s.node.shape_env.size_hint(s.node.expr) if isinstance(s, torch.SymInt) else s for s in x.size()]
        stride = [s.node.shape_env.size_hint(s.node.expr) if isinstance(s, torch.SymInt) else s for s in x.stride()]
    else:
        size = x.size()
        stride = x.stride()
    y = torch.empty_strided(size, stride, dtype=x.dtype, device=x.device, requires_grad=x.requires_grad)
    y.zero_()
    return y