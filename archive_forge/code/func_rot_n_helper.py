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
@lru_cache(32)
def rot_n_helper(n):
    assert n > 1
    vars = [f'v{i}' for i in range(n)]
    rotated = reversed(vars[-1:] + vars[:-1])
    fn = eval(f'lambda {','.join(vars)}: ({','.join(rotated)})')
    fn.__name__ = f'rot_{n}_helper'
    return fn