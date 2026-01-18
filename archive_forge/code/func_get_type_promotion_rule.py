from __future__ import annotations
import abc
import dataclasses
import inspect
import logging
from types import ModuleType
from typing import Any, Callable, Mapping, Optional, Sequence, Set
import torch
import torch._ops
import torch.fx
import torch.fx.traceback as fx_traceback
from torch import _prims_common, _refs
from torch._prims_common import (
from torch._refs import linalg as _linalg_refs, nn as _nn_refs, special as _special_refs
from torch._refs.nn import functional as _functional_refs
from torch._subclasses import fake_tensor
from torch.fx.experimental import proxy_tensor
from torch.fx.node import Node  # noqa: F401
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass, diagnostics, type_utils as fx_type_utils
from torch.utils import _python_dispatch, _pytree
@_beartype.beartype
def get_type_promotion_rule(diagnostic: diagnostics.Diagnostic, node: torch.fx.Node, type_promotion_table: TypePromotionTable) -> Optional[TypePromotionRule]:
    """Get type promotion rule for a node.

    Args:
        diagnostic: Diagnostic object.
        node: Node to get type promotion rule for.
        type_promotion_table: Type promotion table.

    Returns:
        Type promotion rule for the node. None if no rule is found or if the node is not
        representing a torch operator.
    """
    op = node.target
    if not isinstance(op, torch._ops.OpOverload):
        diagnostic.message = f'Skipped for {diagnostics.format_argument(node)}: node.target is not OpOverload. Got type: {type(op)}'
        return None
    if (rule := type_promotion_table.get_rule(op.overloadpacket)) is None:
        diagnostic.message = f'Skipped for {diagnostics.format_argument(node)}: Cannot find type promotion rule for op: {op}'
        return None
    diagnostic.info('Found type promotion rule: %s', rule)
    return rule