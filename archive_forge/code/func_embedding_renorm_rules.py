import torch
from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import (
@register_prop_rule(aten.embedding_renorm_.default)
def embedding_renorm_rules(op_schema: OpSchema) -> OutputSharding:
    raise NotImplementedError('DTensor does not support sharded embedding operation with max_norm yet!')