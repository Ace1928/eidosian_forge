import logging
import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union
import torch
from torch.distributed import is_available
def create_child_mesh(self, device_mesh: 'DeviceMesh', mesh_dim: int, mesh_dim_name: str) -> 'DeviceMesh':
    cur_rank = device_mesh.get_rank()
    pg_ranks_by_dim = device_mesh.mesh.swapdims(-1, mesh_dim).reshape(-1, device_mesh.mesh.size(mesh_dim))
    for mesh_1d in pg_ranks_by_dim:
        sub_mesh = DeviceMesh(device_mesh.device_type, mesh_1d, mesh_dim_names=(mesh_dim_name,), _init_process_groups=False)
        if cur_rank in mesh_1d:
            res_sub_mesh = sub_mesh
    res_sub_mesh._dim_group_infos = [device_mesh._dim_group_infos[mesh_dim]]
    self.child_to_parent_mapping[res_sub_mesh] = device_mesh
    return res_sub_mesh