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
def dynamo_timed(original_function=None, phase_name=None):

    def dynamo_timed_inner(func):
        if config.cprofile:
            return func

        @wraps(func)
        def time_wrapper(*args, **kwargs):
            key = func.__qualname__
            if key not in compilation_time_metrics:
                compilation_time_metrics[key] = []
            with torch.profiler.record_function(f'{key} (dynamo_timed)'):
                t0 = time.time()
                r = func(*args, **kwargs)
                time_spent = time.time() - t0
            compilation_time_metrics[key].append(time_spent)
            if phase_name:
                frame_key = str(curr_frame)
                if frame_key not in frame_phase_timing:
                    frame_phase_timing[frame_key] = {}
                assert phase_name not in frame_phase_timing[frame_key], f'Duplicate phase name {phase_name} for frame {frame_key}'
                frame_phase_timing[frame_key][phase_name] = time_spent
            return r
        return time_wrapper
    if original_function:
        return dynamo_timed_inner(original_function)
    return dynamo_timed_inner