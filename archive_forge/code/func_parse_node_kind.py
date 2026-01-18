from __future__ import annotations
import dataclasses
import re
import typing
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union
import torch
from torch import _C
from torch._C import _onnx as _C_onnx
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, registration
@_beartype.beartype
def parse_node_kind(kind: str) -> Tuple[str, str]:
    """Parse node kind into domain and Op name."""
    if '::' not in kind:
        raise ValueError(f"Node kind: {kind} is invalid. '::' is not in node kind.")
    domain, opname = kind.split('::', 1)
    if '::' in opname:
        raise ValueError(f"Node kind: {kind} is invalid. '::' should only apear once.")
    return (domain, opname)