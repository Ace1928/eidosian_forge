from typing import cast, Dict, List, Optional, Tuple
import torch
from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.ops.utils import prod
from torch.distributed._tensor.placement_types import DTensorSpec, TensorMeta
def pointwise_rule(op_schema: OpSchema, linearity: bool=False) -> OutputSharding:
    """
    Propagate the sharding for pointwise operations.

    Examples:
        ij,ij->ij - addition/mul
        ij,j->ij - broadcasted addition
    """
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    input_specs = op_schema.args_spec
    max_dim = max((input.ndim for input in input_specs))
    dimchars = []
    singleton_counter: List[int] = [0] * max_dim
    for input in input_specs:
        start_dim = max_dim - input.ndim
        p = alphabet[start_dim:max_dim]
        if len(input_specs) > 1:
            for i in range(max_dim):
                if i < start_dim:
                    singleton_counter[i] += 1
                elif input.shape[i - start_dim] == 1:
                    singleton_counter[i] += 1
                    p = _replace_char_in_str(p, '1', i - start_dim)
        dimchars.append(p)
    out_dimchars = alphabet[:max_dim]
    for output_dim_idx in range(len(out_dimchars)):
        out_dimchar = out_dimchars[output_dim_idx]
        if singleton_counter[output_dim_idx] == len(input_specs):
            out_dimchars = _replace_char_in_str(out_dimchars, '1', output_dim_idx)
    fmt = f'{','.join((p for p in dimchars))}->{out_dimchars}'
    enforce_sharding: Dict[str, int] = {}
    if _is_inplace_op(op_schema.op):
        for out_dimchar, mesh_dim in zip(out_dimchars, input_specs[0].dim_map):
            enforce_sharding[out_dimchar] = mesh_dim
    elif _is_out_variant_op(op_schema.op):
        out_spec = cast(DTensorSpec, op_schema.kwargs_schema['out'])
        for out_dimchar, mesh_dim in zip(out_dimchars, out_spec.dim_map):
            enforce_sharding[out_dimchar] = mesh_dim
    return einop_rule(fmt, op_schema, linearity=linearity, enforce_sharding=enforce_sharding)