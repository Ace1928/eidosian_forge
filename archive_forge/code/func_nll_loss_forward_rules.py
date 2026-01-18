from typing import List
import numpy as np
import torch
from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import DTensorSpec, TensorMeta
@register_prop_rule(aten.nll_loss_forward.default)
def nll_loss_forward_rules(op_schema: OpSchema) -> OutputSharding:
    input_spec = op_schema.args_schema[0]
    assert isinstance(input_spec, DTensorSpec)
    assert input_spec.tensor_meta is not None
    result_shape: List[int] = []
    result_stride: List[int] = []
    result_dim = 0
    total_weight_shape: List[int] = []
    total_weight_stride: List[int] = []
    total_weight_dim = 0
    result_tensor_meta = TensorMeta(torch.Size(result_shape), tuple(result_stride), input_spec.tensor_meta.dtype)
    total_weight_tensor_meta = TensorMeta(torch.Size(total_weight_shape), tuple(result_stride), input_spec.tensor_meta.dtype)
    result_spec = DTensorSpec.from_dim_map(input_spec.mesh, [-1 for _ in range(result_dim)], [], tensor_meta=result_tensor_meta)
    total_weight_spec = DTensorSpec.from_dim_map(input_spec.mesh, [-1 for _ in range(total_weight_dim)], [], tensor_meta=total_weight_tensor_meta)
    return OutputSharding([result_spec, total_weight_spec])