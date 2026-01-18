from enum import Enum
from typing import Optional, Tuple
import torch
from torch import Tensor
def random_sparse_mask(dense: Tensor, percent: float, dim: int) -> Tensor:
    """Get a random sparse mask

    Args:
        dense (Tensor):
            Input dense tensor (no zeros).
        percent (float):
            Percent of non-zeros (0, 100].
        dim (int):
            Dimension on which the random sparse mask is computed.
    """
    assert percent > 0 and percent <= 100, percent
    rand = torch.rand_like(dense)
    ones = torch.ones_like(dense)
    k = _get_k_for_topk(percent, None, dense.shape[dim])
    return _scatter_topk_to_sparse_tensor(rand, ones, k, dim)