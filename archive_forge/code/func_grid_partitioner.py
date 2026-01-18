import math
import os
import torch
import weakref
from functools import lru_cache
from torch.utils._triton import has_triton
from ._triton_ops_meta import get_meta
from typing import Optional, Tuple
def grid_partitioner(full_grid, grid_blocks, tensor_dims_map):
    assert 0 <= len(full_grid) <= 3
    assert 0 <= len(grid_blocks) <= 3
    import itertools

    def generate_grid_points():
        for fg, mg in zip(full_grid, grid_blocks):
            yield range(0, fg, mg)

    def generate_sliced_tensors(slices):
        for t, t_dims in tensor_dims_map.items():
            yield next(multidim_slicer(t_dims, slices, t))
    for grid_point in itertools.product(*generate_grid_points()):
        grid = [min(fg - gp, mg) for fg, gp, mg in zip(full_grid, grid_point, grid_blocks)]
        slices = [slice(gp, gp + g) for gp, g in zip(grid_point, grid)]
        yield (grid[::-1], *generate_sliced_tensors(slices))