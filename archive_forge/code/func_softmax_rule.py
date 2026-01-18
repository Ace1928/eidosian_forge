from typing import cast, List, Optional, Sequence, Tuple
import torch
import torch.distributed.distributed_c10d as c10d
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import (
from torch.distributed._tensor.placement_types import (
from torch.distributed.device_mesh import DeviceMesh
@register_prop_rule([aten._log_softmax.default, aten._softmax.default], schema_info=RuntimeSchemaInfo(1))
def softmax_rule(op_schema: OpSchema) -> OutputSharding:
    input_spec, softmax_dim, _ = op_schema.args_schema
    input_spec = cast(DTensorSpec, input_spec)
    softmax_dim = cast(int, softmax_dim)
    dim_map = input_spec.dim_map
    if softmax_dim < len(dim_map) and dim_map[softmax_dim] >= 0:
        raise RuntimeError('Cannot run softmax on sharding dimension!')
    return OutputSharding(input_spec)