import torch
from torch.distributed._tensor.op_schema import OpSchema, OpStrategy, OutputSharding
from torch.distributed._tensor.ops.basic_strategy import gen_einsum_strategies
from torch.distributed._tensor.ops.common_rules import einop_rule
from torch.distributed._tensor.ops.utils import (
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.distributed.device_mesh import DeviceMesh
@register_op_strategy(aten.bmm.default)
def bmm_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    return _mm_like_strategy('bmk,bkn->bmn', mesh, op_schema)