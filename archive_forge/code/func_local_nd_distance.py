import math
from typing import List
import numpy as np
import torch
from xformers.components.attention.sparsity_config import (
def local_nd_distance(*sizes, p=2.0, weights=None):
    if weights is None:
        weights = (1,) * len(sizes)
    assert len(sizes) == len(weights)
    grid = _generate_nd_grid(*sizes)
    grid = [i.flatten() * w for i, w in zip(grid, weights)]
    grid = torch.stack(grid, dim=1).float()
    d = torch.cdist(grid, grid, p=p)
    return d