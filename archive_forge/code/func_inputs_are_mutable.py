import collections
import dataclasses
import enum
import itertools as it
import logging
from typing import (
from typing_extensions import Literal
import torch
from torch._C import FunctionSchema
from torch._C._autograd import _ProfilerResult
from torch._C._profiler import (
from torch._utils import _element_size
from torch.profiler import _utils
@classmethod
def inputs_are_mutable(cls, t: _ExtraFields_TorchOp) -> Tuple[Optional[bool], ...]:
    """Determine which inputs may have mutated based on function schema.

        Note that we don't need to resolve down to a single schema to perform
        this analysis. An input is mutable if it is mutable in any overload. In
        practice, however, it is overwhelmingly common to match a single
        overload. If we cannot find any valid schema then we must be
        conservative and assume all inputs are mutable.
        """
    mutable: Optional[List[bool]] = None
    for schema in cls.match_schemas(t):
        mutable = mutable or [False for _ in schema.arguments]
        for i, arg in enumerate(schema.arguments):
            mutable[i] |= getattr(arg.alias_info, 'is_write', False)
    return tuple(mutable or (None for _ in t.inputs))