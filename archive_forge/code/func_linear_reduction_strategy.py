from typing import cast, List, Optional, Sequence, Tuple
import torch
import torch.distributed.distributed_c10d as c10d
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import (
from torch.distributed._tensor.placement_types import (
from torch.distributed.device_mesh import DeviceMesh
@register_op_strategy(list(LINEAR_REDUCTION_OP_MAP.keys()), schema_info=RuntimeSchemaInfo(1))
def linear_reduction_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    args_schema = op_schema.args_schema
    input_strategy = args_schema[0]
    assert isinstance(input_strategy, OpStrategy)
    dims = None
    if len(op_schema.args_schema) > 1:
        dims = _infer_reduction_dims(args_schema[1], input_strategy.output_ndim)
    reduce_dims = list(range(input_strategy.output_ndim)) if dims is None else dims
    keep_dim = len(op_schema.args_schema) > 2 and bool(op_schema.args_schema[2])
    reduction_op = LINEAR_REDUCTION_OP_MAP[op_schema.op]
    return common_reduction_strategy(mesh, input_strategy, reduce_dims, keep_dim=keep_dim, reduction_linear=True, reduction_op=reduction_op)