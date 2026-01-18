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
def tensor_always_has_static_shape(tensor: Union[torch.Tensor, Any], is_tensor: bool, guard_source: 'torch._guards.GuardSource') -> Tuple[bool, Optional[TensorStaticReason]]:
    """
    Given a tensor, source, and is_tensor flag, determine if a shape should be static.

    Args:
    tensor - the real tensor to evaluate, parameters force a static shape.
    is_tensor - internal dynamo check, essentially "is_tensor": target_cls is TensorVariable,
    tensors not in a TensorVariable for whatever reason are forced static.

    Returns a tuple, where the first element is the bool of whether or not this tensor should have a static shape.
    The second element is a TensorStaticReason, useful for passing to tensor_static_reason_to_message if needed.
    """
    if guard_source.is_nn_module() and config.force_nn_module_property_static_shapes:
        return (True, TensorStaticReason.NN_MODULE_PROPERTY)
    if type(tensor) is torch.nn.Parameter and config.force_parameter_static_shapes:
        return (True, TensorStaticReason.PARAMETER)
    if not is_tensor:
        return (True, TensorStaticReason.NOT_TENSOR)
    return (False, None)