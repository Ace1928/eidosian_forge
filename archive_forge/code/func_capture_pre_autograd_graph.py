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
@compatibility(is_backward_compatible=False)
def capture_pre_autograd_graph(f: Callable, args: Tuple[Any], kwargs: Optional[Dict[str, Any]]=None, constraints: Optional[List[Constraint]]=None) -> torch.nn.Module:
    """
    A helper function that is intended to trace a module before any pre-autograd
    decomposition is run. The produced module will be "non-functional" and
    composed of aten operators. Later this API will be deleted in favor of more general
    torch.export API.

    Args:
      f: A callable to be traced

      args: example positional inputs.

      kwargs: optional example keyword inputs.

      constraints: A optional list of constraints on the dynamic arguments specifying
            their possible range of their shapes

    Returns:
        An nn.Module containing the traced method.

    """
    decomp_table = {torch.ops.aten.dropout.default: torch.ops.aten.dropout.default.decompose, torch.ops.aten.batch_norm.default: torch.ops.aten.batch_norm.default.decompose, torch.ops.aten._batch_norm_impl_index.default: torch.ops.aten._batch_norm_impl_index.default.decompose, torch.ops.aten.native_batch_norm.default: torch.ops.aten.native_batch_norm.default.decompose}
    if kwargs is None:
        kwargs = {}
    with torch._dynamo.config.patch(dataclasses.asdict(DEFAULT_EXPORT_DYNAMO_CONFIG)):
        m = torch._dynamo.export(f, constraints=constraints, assume_static_by_default=True, tracing_mode='symbolic', decomposition_table=decomp_table, pre_dispatch=True, aten_graph=True)(*args, **kwargs)[0]

        def _train(self, mode: bool=True):
            raise NotImplementedError('Calling train() is not supported yet.')

        def _eval(self, mode: bool=True):
            raise NotImplementedError('Calling eval() is not supported yet.')
        _, _, _, fake_mode = _convert_input_to_fake(m, args, kwargs)
        m.meta['inline_constraints'] = {k: v for k, v in fake_mode.shape_env.runtime_var_to_range.items() if re.match('^[if]\\d+$', str(k))}
        flat_args, _ = pytree.tree_flatten((args, kwargs or {}))
        range_constraints, equality_constraints = _process_constraints(m, 0, flat_args)
        unlifted_m = _create_stateful_graph_module(m, range_constraints=range_constraints, equality_constraints=equality_constraints)
        unlifted_m.train = types.MethodType(_train, m)
        unlifted_m.eval = types.MethodType(_eval, m)
        return unlifted_m