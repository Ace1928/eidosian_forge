import torch
from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import (
@register_prop_rule(aten.embedding.default)
def embedding_rules(op_schema: OpSchema) -> OutputSharding:
    weight_spec, inp_spec = op_schema.args_spec
    if any((placement.is_shard(0) for placement in weight_spec.placements)):
        raise NotImplementedError('DTensor does not support row-wise sharded embedding operation yet!')
    if weight_spec.is_replicated() and inp_spec.placements == [Shard(0)]:
        return OutputSharding(output_spec=DTensorSpec(mesh=inp_spec.mesh, placements=inp_spec.placements))
    if inp_spec.is_replicated():
        weight_dim_map = weight_spec.dim_map
        output_dim_map = inp_spec.dim_map
        output_dim_map.append(weight_dim_map[1])
        return OutputSharding(output_spec=DTensorSpec.from_dim_map(inp_spec.mesh, output_dim_map, []))
    return OutputSharding(output_spec=None, schema_suggestions=[OpSchema(op=op_schema.op, args_schema=(weight_spec, DTensorSpec(mesh=inp_spec.mesh, placements=tuple([Replicate()] * len(inp_spec.placements)), tensor_meta=inp_spec.tensor_meta)), kwargs_schema=op_schema.kwargs_schema)])