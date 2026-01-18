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
@contextlib.contextmanager
def restore_guarded_dynamo_config(first_ctx: bool, saved_config_and_hash: ConfigAndHash, nopython: bool):
    _maybe_init_guarded_config_cache()
    is_top_level = False
    try:
        if first_ctx and config_cache.saved_config_and_hash is None:
            assert config_cache.nopython is None
            is_top_level = True
            config_cache.saved_config_and_hash = saved_config_and_hash
            config_cache.nopython = nopython
            log.debug('Setting top-level compile config hash: %s', saved_config_and_hash.hash.hex())
        else:
            log.debug('Ignoring inner dynamo compile config and hash')
        yield
    finally:
        if is_top_level:
            log.debug('Unsetting top-level compile config hash: %s', config_cache.saved_config_and_hash.hash.hex())
            config_cache.saved_config_and_hash = None
            config_cache.nopython = None