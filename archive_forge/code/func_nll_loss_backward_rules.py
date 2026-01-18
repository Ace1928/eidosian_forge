from typing import List
import numpy as np
import torch
from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import DTensorSpec, TensorMeta
@register_prop_rule(aten.nll_loss_backward.default)
def nll_loss_backward_rules(op_schema: OpSchema) -> OutputSharding:
    input_spec = op_schema.args_schema[1]
    assert isinstance(input_spec, DTensorSpec)
    return OutputSharding(input_spec)