import base64
import dataclasses
import io
import json
import logging
import math
import operator
import typing
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Union
import sympy
import torch
import torch.export.exported_program as ep
from torch._export.verifier import load_verifier
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental import symbolic_shapes
from torch.utils._pytree import treespec_dumps, treespec_loads
from torch.utils._sympy.value_ranges import ValueRanges
from .schema import (  # type: ignore[attr-defined]
from torch.export.exported_program import (
from .upgrade import GraphModuleOpUpgrader
def serialize_sym_int(s: Union[int, torch.SymInt]) -> SymInt:
    if isinstance(s, (torch.SymInt, int)):
        if symbolic_shapes.is_concrete_int(s):
            return SymInt.create(as_int=int(s))
        else:
            assert isinstance(s, torch.SymInt)
            if s.node.hint is None:
                return SymInt.create(as_expr=SymExpr(str(s)))
            else:
                return SymInt.create(as_expr=SymExpr(str(s), hint=SymExprHint.create(as_int=s.node.hint)))
    else:
        raise SerializeError(f'SymInt should be either symbol or int, got `{s}` of type `{type(s)}`')