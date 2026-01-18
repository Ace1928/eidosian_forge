from typing import cast, List, Optional, Sequence, Tuple
import torch
import torch.distributed.distributed_c10d as c10d
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import (
from torch.distributed._tensor.placement_types import (
from torch.distributed.device_mesh import DeviceMesh
@register_op_strategy([aten.native_layer_norm.default], schema_info=RuntimeSchemaInfo())
def layer_norm_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    assert len(op_schema.args_schema) == 5
    input_strategy, normalized_shape, weight_strategy, bias_strategy, _ = op_schema.args_schema
    assert isinstance(input_strategy, OpStrategy)
    assert isinstance(normalized_shape, (int, Sequence, torch.Size))
    normalized_size = normalize_to_torch_size(normalized_shape)
    input_ndim = input_strategy.output_ndim
    axis = input_ndim - len(normalized_size)
    output_strategy = OpStrategy([])
    for idx, input_placement_strategy in enumerate(input_strategy.strategies):
        op_args_target_specs = []
        redistribute_costs = []
        input_src_spec = input_placement_strategy.output_spec
        input_target_spec = DTensorSpec(mesh=mesh, placements=_replicate_dims_start_at(input_src_spec.placements, axis), tensor_meta=input_src_spec.tensor_meta)
        op_args_target_specs.append(input_target_spec)
        redistribute_costs.append(generate_redistribute_costs(input_strategy, input_target_spec))
        if weight_strategy is not None:
            assert isinstance(weight_strategy, OpStrategy)
            weight_src_spec = weight_strategy.strategies[idx].output_spec
            weight_target_spec = DTensorSpec(mesh=mesh, placements=_replicate_dims_start_at(weight_src_spec.placements), tensor_meta=weight_src_spec.tensor_meta)
            op_args_target_specs.append(weight_target_spec)
            redistribute_costs.append(generate_redistribute_costs(weight_strategy, weight_target_spec))
        if bias_strategy is not None:
            assert isinstance(bias_strategy, OpStrategy)
            bias_src_spec = bias_strategy.strategies[idx].output_spec
            bias_target_spec = DTensorSpec(mesh=mesh, placements=_replicate_dims_start_at(bias_src_spec.placements), tensor_meta=bias_src_spec.tensor_meta)
            op_args_target_specs.append(bias_target_spec)
            redistribute_costs.append(generate_redistribute_costs(bias_strategy, bias_target_spec))
        output_target_spec = input_target_spec
        output_strategy.strategies.append(PlacementStrategy(output_spec=output_target_spec, input_specs=op_args_target_specs, redistribute_cost=redistribute_costs))
    return output_strategy