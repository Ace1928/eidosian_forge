from typing import cast, Dict, List, Tuple
import torch
import torch.distributed._tensor.api as dtensor
from torch.distributed._tensor.placement_types import (
from torch.distributed.device_mesh import DeviceMesh
def redistribute_local_tensor(local_tensor: torch.Tensor, current_spec: DTensorSpec, target_spec: DTensorSpec) -> torch.Tensor:
    """
    This redistribute the local tensor (torch.Tensor) from the current DTensorSpec to
    the target DTensorSpec, which involves the necessary collective calls to transform
    the local shard of the DTensor from its current spec to the target spec.
    """
    if current_spec.mesh != target_spec.mesh:
        raise NotImplementedError('Cross device mesh comm not supported yet!')
    new_local_tensor = None
    current_placements = current_spec.placements
    target_placements = target_spec.placements
    sorted_placements = list(enumerate(zip(current_placements, target_placements)))
    sorted_placements = _decompose_reshard(sorted_placements)
    sorted_placements.sort(key=_replicate_then_shard)
    device_mesh = current_spec.mesh
    for i, (current, target) in sorted_placements:
        my_coordinate = device_mesh.get_coordinate()
        num_chunks = device_mesh.size(mesh_dim=i)
        if my_coordinate is None:
            return local_tensor
        if current == target:
            new_local_tensor = local_tensor
            continue
        if target.is_replicate():
            if current.is_partial():
                partial_spec = cast(_Partial, current)
                new_local_tensor = partial_spec._to_replicate(local_tensor, device_mesh, i)
            elif current.is_shard():
                current_placement = cast(Shard, current)
                new_local_tensor = current_placement._to_replicate_tensor(local_tensor, current_spec.shape, device_mesh, i)
            else:
                raise RuntimeError(f'redistribute from {current_placements} to {target_placements} not supported yet')
        elif target.is_shard():
            target_placement = cast(Shard, target)
            if current.is_partial():
                partial_spec = cast(_Partial, current)
                new_local_tensor = partial_spec._to_shard(local_tensor, device_mesh, i, target_placement)
            elif current.is_replicate():
                shards, _ = target_placement._split_tensor(local_tensor, num_chunks, with_padding=False, contiguous=False)
                new_local_tensor = shards[my_coordinate[i]].clone()
            else:
                assert current.is_shard(), f'Current placement should be shard but found {current}'
                shard_spec = cast(Shard, current)
                if shard_spec.dim != target_placement.dim:
                    raise NotImplementedError('Changing sharding dim is not supported yet!')
        elif target.is_partial():
            if current.is_replicate():
                new_local_tensor = local_tensor / num_chunks
            else:
                raise RuntimeError(f'redistribute from {current_placements} to {target_placements} not supported yet')
        assert new_local_tensor is not None
        local_tensor = new_local_tensor
    assert new_local_tensor is not None, 'redistribute failed!'
    return new_local_tensor