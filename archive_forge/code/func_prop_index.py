from typing import cast, List, Optional, Sequence, Tuple
import torch
from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import (
from torch.distributed._tensor.placement_types import (
from torch.distributed.device_mesh import DeviceMesh
@register_prop_rule(aten.index.Tensor, schema_info=RuntimeSchemaInfo(needs_pytree=True))
def prop_index(op_schema: OpSchema) -> OutputSharding:
    """
    Expect replicated on the first input; _mostly_ pointwise on the second input.

    TODO: exception: when the dtype of second input is "bool", then a torch.nonzero needs to be triggered first.
    """
    values_spec, multi_indices_spec = op_schema.args_schema
    assert isinstance(values_spec, DTensorSpec)
    assert isinstance(multi_indices_spec, list)
    multi_indices_spec = cast(List[Optional[DTensorSpec]], multi_indices_spec)
    valid_indices_spec: List[Tuple[int, DTensorSpec]] = [(i, a) for i, a in enumerate(multi_indices_spec) if a is not None]
    indices_out = pointwise_rule(OpSchema(op=op_schema.op, args_schema=tuple((v[1] for v in valid_indices_spec)), kwargs_schema={}))
    need_reshard_on_indices = indices_out.output_spec is None
    if not need_reshard_on_indices:
        assert isinstance(indices_out.output_spec, DTensorSpec)
        indices_spec: DTensorSpec = indices_out.output_spec
    else:
        assert indices_out.schema_suggestions is not None
        valid_indices_suggestion = indices_out.schema_suggestions[0]
        for i, v in enumerate(valid_indices_suggestion.args_spec):
            multi_indices_spec[valid_indices_spec[i][0]] = v
        indices_output_spec = pointwise_rule(valid_indices_suggestion).output_spec
        assert isinstance(indices_output_spec, DTensorSpec)
        indices_spec = indices_output_spec
    lookup_dims = {v[0] for v in valid_indices_spec}
    need_reshard_on_values = tuple((isinstance(vp, Shard) and (vp.dim in lookup_dims or isinstance(ip, Shard)) for vp, ip in zip(values_spec.placements, indices_spec.placements)))
    if not need_reshard_on_indices and (not any(need_reshard_on_values)):
        value_placements = values_spec.placements
        all_dims_consecutive = all((b[0] - a[0] == 1 for b, a in zip(valid_indices_spec[1:], valid_indices_spec[:-1])))
        if all_dims_consecutive:
            insert_dim: int = valid_indices_spec[0][0]
        else:
            insert_dim = 0

        def place(vp: Placement, ip: Placement) -> Placement:
            if isinstance(vp, Shard):
                return Shard(vp.dim if vp.dim < insert_dim else vp.dim + indices_spec.ndim - sum((1 if vp.dim > v[0] else 0 for v in valid_indices_spec)))
            if isinstance(ip, Shard):
                return Shard(ip.dim + insert_dim)
            return vp
        value_placements = tuple((place(vp, ip) for vp, ip in zip(values_spec.placements, indices_spec.placements)))
        result = OutputSharding(output_spec=DTensorSpec(mesh=values_spec.mesh, placements=value_placements))
        return result
    else:
        result = OutputSharding(output_spec=None, schema_suggestions=[OpSchema(op=op_schema.op, args_schema=(DTensorSpec(mesh=values_spec.mesh, placements=tuple([Replicate() if need_reshard_on_values[i] else v for i, v in enumerate(values_spec.placements)]), tensor_meta=values_spec.tensor_meta), multi_indices_spec), kwargs_schema=op_schema.kwargs_schema)])
        return result