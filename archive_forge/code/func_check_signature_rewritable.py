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
def check_signature_rewritable(graph):
    input_errors = []
    for node in graph.graph.nodes:
        if node.op == 'placeholder':
            assert hasattr(node, '_dynamo_source')
            source = node._dynamo_source
            user_stacks = graph._source_to_user_stacks.get(source)
            if user_stacks is None:
                continue
            assert len(user_stacks) > 0
            stack = None
            for s in user_stacks:
                if len(s) == 0:
                    continue
                stack = s
                break
            if stack is None:
                msg = f'{source.name()}, a closed over free variable'
            else:
                tb = ''.join(traceback.format_list(stack))
                extra = ''
                if len(user_stacks) > 1:
                    extra = f'(elided {len(user_stacks) - 1} more accesses)'
                msg = f'{source.name()}, accessed at:\n{tb}{extra}'
            input_errors.append(msg)
    if input_errors:
        raise UserError(UserErrorType.INVALID_INPUT, "Cannot export model which references tensors that are neither buffers/parameters/constants nor are direct inputs.  For each tensor, if you'd like this tensor to be an explicit input, add it as a dummy argument to the top-level model definition you are exporting; if you would like its value to be embedded as an exported constant, wrap its access in a function marked with @assume_constant_result.\n\n" + '\n\n'.join(input_errors))