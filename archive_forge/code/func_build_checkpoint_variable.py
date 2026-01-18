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
def build_checkpoint_variable(**options):
    import torch._higher_order_ops.wrap as higher_order_ops
    from .variables.higher_order_ops import TorchHigherOrderOperatorVariable
    activation_checkpoint_op: 'torch._ops.HigherOrderOperator' = higher_order_ops.tag_activation_checkpoint
    if torch._functorch.config.functionalize_rng_ops:
        activation_checkpoint_op = higher_order_ops.wrap_activation_checkpoint
    return TorchHigherOrderOperatorVariable.make(activation_checkpoint_op, **options)