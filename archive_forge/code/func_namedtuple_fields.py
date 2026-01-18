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
@functools.lru_cache(1)
def namedtuple_fields(cls):
    """Get the fields of a namedtuple or a torch.return_types.* quasi-namedtuple"""
    if cls is slice:
        return ['start', 'stop', 'step']
    assert issubclass(cls, tuple)
    if hasattr(cls, '_fields'):
        return cls._fields

    @dataclasses.dataclass
    class Marker:
        index: int
    assert cls.__module__ == 'torch.return_types'
    obj = cls(map(Marker, range(cls.n_fields)))
    fields: List[Optional[str]] = [None] * cls.n_fields
    for name in dir(obj):
        if name[0] != '_' and isinstance(getattr(obj, name), Marker):
            fields[getattr(obj, name).index] = name
    return fields