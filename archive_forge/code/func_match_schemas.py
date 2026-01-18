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
def match_schemas(cls, t: _ExtraFields_TorchOp) -> Tuple[FunctionSchema, ...]:
    signature = tuple((TensorKey.from_tensor(i) if isinstance(i, _TensorMetadata) else [TensorKey.from_tensor(j) for j in i] if isinstance(i, list) else i for i in t.inputs))

    def matches(schema) -> bool:
        return len(schema.arguments) == len(signature) and all((cls._types_match(observed, schema_arg.type) for observed, schema_arg in zip(signature, schema.arguments)))
    return tuple((s for s in cls.lookup_schemas(t.name) or () if matches(s)))