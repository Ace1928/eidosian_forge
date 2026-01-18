from typing import List
import torch
from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import DTensorSpec, TensorMeta
@register_prop_rule(aten.convolution_backward.default)
def convolution_backward_rules(op_schema: OpSchema) -> OutputSharding:
    input_spec = op_schema.args_schema[0]
    grad_output_spec, input_spec, weight_spec, bias_shape_opt, stride, padding, dilation, transposed, output_padding, groups, output_mask = op_schema.args_schema
    assert isinstance(grad_output_spec, DTensorSpec)
    assert isinstance(input_spec, DTensorSpec)
    assert isinstance(weight_spec, DTensorSpec)
    assert isinstance(bias_shape_opt, List)
    assert input_spec.tensor_meta is not None
    weight_tensor_meta = weight_spec.tensor_meta
    bias_tensor_meta = TensorMeta(torch.Size(bias_shape_opt), (1,), input_spec.tensor_meta.dtype)
    grad_input_spec = input_spec
    grad_weight_spec = DTensorSpec.from_dim_map(input_spec.mesh, [-1, -1, -1, -1], [0], tensor_meta=weight_tensor_meta)
    grad_bias_spec = DTensorSpec.from_dim_map(input_spec.mesh, [-1], [0], tensor_meta=bias_tensor_meta)
    return OutputSharding([grad_input_spec, grad_weight_spec, grad_bias_spec])