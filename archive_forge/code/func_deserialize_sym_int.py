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
def deserialize_sym_int(self, s: SymInt) -> Union[int, torch.SymInt]:
    val = s.value
    if s.type == 'as_expr':
        if val.expr_str in self.symbol_name_to_symbol:
            sym = self.symbol_name_to_symbol[val.expr_str]
        else:
            sym = sympy.sympify(val.expr_str, locals=self.symbol_name_to_symbol)
            if isinstance(sym, sympy.Symbol):
                self.symbol_name_to_symbol[val.expr_str] = sym
                if (vr := self.symbol_name_to_range.get(val.expr_str)):
                    symbolic_shapes._constrain_symbol_range(self.shape_env, sym, compiler_min=vr.lower, compiler_max=vr.upper, runtime_min=vr.lower, runtime_max=vr.upper)
        if val.hint is None:
            hint = None
        else:
            assert val.hint.type == 'as_int'
            hint = val.hint.value
        return self.shape_env.create_symintnode(sym, hint=hint)
    elif s.type == 'as_int':
        assert isinstance(val, int)
        return val
    else:
        raise SerializeError(f'SymInt has invalid field type {s.type} with value {s.value}')