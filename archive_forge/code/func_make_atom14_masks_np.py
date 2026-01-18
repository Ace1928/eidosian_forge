from typing import Dict
import numpy as np
import torch
from . import residue_constants as rc
from .tensor_utils import tensor_tree_map, tree_map
def make_atom14_masks_np(batch: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    batch = tree_map(lambda n: torch.tensor(n, device=batch['aatype'].device), batch, np.ndarray)
    out = tensor_tree_map(lambda t: np.array(t), make_atom14_masks(batch))
    return out