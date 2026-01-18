import collections
import contextlib
import copy
import dataclasses
import dis
import functools
import importlib
import inspect
import itertools
import linecache
import logging
import operator
import sys
import textwrap
import threading
import traceback
import types
import typing
import weakref
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Type
from unittest.mock import patch
import torch
import torch._logging
from torch._guards import Checkpointable, tracing, TracingContext
from . import (
from .allowed_functions import is_allowed, is_builtin_constant, is_forbidden
from .bytecode_analysis import (
from .bytecode_transformation import (
from .code_context import code_context
from .codegen import PyCodegen
from .current_scope_id import current_scope_id
from .exc import ArgsMismatchError, BackendCompilerFailed, unimplemented, Unsupported
from .funcname_cache import get_funcname
from .guards import GuardBuilder, install_guard
from .output_graph import GraphCompileReason, OutputGraph, OutputGraphState
from .replay_record import DummyModule, ExecutionRecorder
from .resume_execution import ContinueExecutionCache, ReenterWith
from .source import (
from .utils import (
from .variables.base import (
from .variables.builder import VariableBuilder, wrap_fx_proxy
from .variables.builtin import BuiltinVariable
from .variables.constant import ConstantVariable, EnumVariable
from .variables.ctx_manager import (
from .variables.dicts import ConstDictVariable, SetVariable
from .variables.functions import (
from .variables.lists import (
from .variables.misc import (
from .variables.nn_module import NNModuleVariable
from .variables.tensor import (
from .variables.torch import TorchVariable
from .variables.user_defined import (
def calc_package(self):
    """
        Copied from the Cpython implementation of __import__
        https://github.com/python/cpython/blob/5a094f0255eea1db58fb2cf14c200971e64ec36e/Lib/importlib/_bootstrap.py#L1090
        """
    package = self.f_globals.get('__package__')
    spec = self.f_globals.get('__spec__')
    if package is not None:
        if spec is not None and package != spec.parent:
            log.warning('__package__ != __spec__.parent (%r != %r)', package, spec.parent, stacklevel=3)
        return package
    elif spec is not None:
        return spec.parent
    else:
        log.warning("can't resolve package from __spec__ or __package__, falling back on __name__ and __path__", stacklevel=3)
        package = self.f_globals['__name__']
        if '__path__' not in self.f_globals:
            package = package.rpartition('.')[0]
    return package