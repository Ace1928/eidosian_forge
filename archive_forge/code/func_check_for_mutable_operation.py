import torch
import inspect
import numbers
import types
import typing
import enum
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, NamedTuple, cast, TYPE_CHECKING
from torch._jit_internal import boolean_dispatched
from ._compatibility import compatibility
from torch._ops import OpOverloadPacket, OpOverload
@compatibility(is_backward_compatible=False)
def check_for_mutable_operation(target: Callable, args: Tuple['Argument', ...], kwargs: Dict[str, 'Argument']):
    signatures, schemas = get_signature_for_torch_op(target, return_schemas=True)
    if signatures and schemas:
        matched_schemas = []
        for candidate_signature, schema in zip(signatures, schemas):
            try:
                candidate_signature.bind(*args, **kwargs)
                matched_schemas.append((candidate_signature, schema))
            except TypeError as e:
                continue

        def throw_if_mutable(schema):
            if schema.is_mutable:
                raise RuntimeError(f'Tried to trace mutable operation {schema}. FX only supports functional code, so operations that mutate operands in-place (e.g. via `out` arguments) are not supported')
        if len(matched_schemas) == 0:
            pass
        elif len(matched_schemas) == 1:
            _, schema_to_check = matched_schemas[0]
            throw_if_mutable(schema_to_check)
            pass
        else:
            pass