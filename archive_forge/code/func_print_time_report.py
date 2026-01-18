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
def print_time_report():
    total = 0.0
    total_by_key = {}
    for timings in frame_phase_timing.values():
        for key, timing in timings.items():
            total += timing
            if key not in total_by_key:
                total_by_key[key] = timing
            else:
                total_by_key[key] += timing
    out = 'TIMING:'
    for key, value in total_by_key.items():
        out = f'{out} {key}:{round(value, 5)}'
    print(out)