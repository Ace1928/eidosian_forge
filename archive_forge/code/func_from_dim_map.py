from dataclasses import dataclass
from typing import Any, cast, List, NamedTuple, Optional, Tuple
import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch.distributed._tensor._collective_utils import mesh_broadcast, mesh_scatter
from torch.distributed.device_mesh import DeviceMesh
@classmethod
def from_dim_map(cls, mesh: DeviceMesh, dim_map: List[int], sums: List[int], tensor_meta: Optional[TensorMeta]=None) -> 'DTensorSpec':
    """
        Construct a DTensorSpec from dim_map list and pending sum.

        Args:
            mesh (class:`DeviceMesh`): device mesh to be used in the DTensorSpec
            dim_map (List[int]): a list of integer that represents sharding on each
                tensor dimension, see `dim_map` property doc for details
            sums (List[int]): a list of integer that represents the dist tensor have
                pending sum on which device mesh dimension.
            tensor meta (TensorMeta): DTensor metadata

        Return:
            a class:`DTensorSpec` object
        """
    placements: List[Placement] = [Replicate() for _ in range(mesh.ndim)]
    for s in sums:
        placements[s] = _Partial()
    for i, m in enumerate(dim_map):
        if m >= 0:
            placement = placements[m]
            if placement.is_shard():
                placement = cast(Shard, placement)
                raise RuntimeError(f"DeviceMesh dimension cann't be mapped to two dimension of the same tensor: {i} and {placement.dim}")
            elif placement.is_partial():
                raise RuntimeError(f'DeviceMesh dimension {m} cannot be both shard and partial!')
            placements[m] = Shard(i)
    return cls(mesh, tuple(placements), tensor_meta=tensor_meta)