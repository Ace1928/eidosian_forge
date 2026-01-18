from __future__ import annotations
import contextlib
import dis
import functools
import inspect
import logging
import os
import sys
import textwrap
import threading
import traceback
import types
import warnings
from dataclasses import dataclass
from enum import Enum
from os.path import dirname, join
from typing import (
from unittest.mock import patch
import torch
import torch.fx
import torch.utils._pytree as pytree
import torch.utils.checkpoint
from torch import _guards
from torch._subclasses import fake_tensor
from torch.export import Constraint
from torch.fx.experimental.proxy_tensor import make_fx, maybe_disable_fake_tensor_mode
from torch.fx.experimental.symbolic_shapes import (
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo
from torch.nn.parallel.distributed import DistributedDataParallel
from ..fx import GraphModule
from .backends.registry import CompilerFn, lookup_backend
from .hooks import Hooks
from . import config, convert_frame, external_utils, skipfiles, utils
from .code_context import code_context
from .exc import CondOpArgsMismatchError, UserError, UserErrorType
from .mutation_guard import install_generation_tagging_init
from .types import CacheEntry, DynamoCallback
from .utils import compile_times
from torch._dispatch.python import enable_python_dispatcher
from torch.utils._python_dispatch import _disable_current_modes
import sympy
def result_capturing_wrapper(*graph_inputs):
    nonlocal graph_captured_result
    nonlocal graph_captured_input
    graph_captured_input = graph_inputs
    assert graph is not None
    named_parameters = dict(graph.named_parameters(remove_duplicate=False))
    named_buffers = dict(graph.named_buffers(remove_duplicate=False))
    ambient_fake_mode = _guards.detect_fake_mode(graph_inputs) if _guards.detect_fake_mode(graph_inputs) is not None else fake_mode
    with ambient_fake_mode, enable_python_dispatcher():
        params_and_buffers = {**dict(named_parameters), **dict(named_buffers)}
        fake_params_buffers = dict()
        for name, value in params_and_buffers.items():
            fake_params_buffers[name] = ambient_fake_mode.from_tensor(value, static_shapes=True)
        fake_graph_inputs = pytree.tree_map(ambient_fake_mode.from_tensor, graph_inputs)
        graph_captured_result = torch.func.functional_call(graph, fake_params_buffers, fake_graph_inputs)
    return graph_captured_result