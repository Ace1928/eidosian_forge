import copy
import dataclasses
import functools
import io
import json
import pathlib
import re
import sys
import types
import warnings
import weakref
import zipfile
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from unittest.mock import patch
import sympy
import torch
import torch._dynamo
import torch.fx
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch._decomp import core_aten_decompositions, get_decompositions
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.exc import UserError, UserErrorType
from torch._dynamo.source import ConstantSource
from torch._export.passes.collect_tracepoints_pass import CollectTracepointsPass
from torch._functorch.aot_autograd import aot_export_module, GraphSignature
from torch._functorch.eager_transforms import functionalize
from torch._guards import detect_fake_mode
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.export import _create_constraint, _Dim, Constraint
from torch.export.exported_program import (
from torch.export.graph_signature import (
from torch.fx import traceback as fx_traceback
from torch.fx._compatibility import compatibility
from torch.fx.experimental.proxy_tensor import make_fx, maybe_disable_fake_tensor_mode
from torch.fx.experimental.symbolic_shapes import (
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo
from torch.utils._sympy.value_ranges import ValueRangeError, ValueRanges
from .exported_program import (
from .passes.add_runtime_assertions_for_constraints_pass import (
from .passes.lift_constant_tensor_pass import lift_constant_tensor_pass
from .passes.remove_runtime_assertions import _RemoveRuntimeAssertionsPass
from .passes.replace_sym_size_ops_pass import _replace_sym_size_ops_pass
from .passes.replace_view_ops_with_view_copy_ops_pass import (
from .wrappers import _wrap_submodules
def export__RC__(f: Callable, args: Tuple[Any, ...], kwargs: Optional[Dict[str, Any]]=None, *, dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]]=None, strict: bool=True, preserve_module_call_signature: Tuple[str, ...]=()) -> ExportedProgram:
    """
    API for exporting with dynamic shape specifications instead of constraints.
    It should be considered "release candidate" (RC), meant to replace `export`.

    Here, `dynamic_shapes` is expected to be a dict from
    argument names of `f` to dynamic shape specifications OR a tuple where each element
    corresponds to the original order of the arguments defined in the function signature
    ,as follows:
    - The dynamic shape of a tensor argument can be specified as:
      - Either a dict from dynamic dimension indices to Dim types. It is not
        required to include static dimension indices in this dict, but when
        they are, they should be mapped to None.
      - Or a tuple of Dim types or None. The Dim types correspond to dynamic
        dimensions, whereas static dimensions are denoted by None.
    - Arguments that are dicts or tuples of tensors are recursively specified
      by using mappings or sequences of contained specifications.

    See `export` for documentation of `f`, `args`, `kwargs` and return.
    """
    constraints = _process_dynamic_shapes(f, args, kwargs, dynamic_shapes)
    return _export(f, args, kwargs, constraints=constraints, strict=strict, preserve_module_call_signature=preserve_module_call_signature)