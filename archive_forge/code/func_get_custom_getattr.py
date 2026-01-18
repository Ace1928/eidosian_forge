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
def get_custom_getattr(value: Any):
    try:
        getattr_fn = inspect.getattr_static(type(value), '__getattr__')
    except AttributeError:
        getattr_fn = None
    if getattr_fn is torch.nn.Module.__getattr__:
        getattr_fn = None
    return getattr_fn