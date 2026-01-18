from typing import Dict, Tuple, overload
import torch
import torch.types
from torch import nn
from . import residue_constants as rc
from .rigid_utils import Rigid, Rotation
from .tensor_utils import batched_gather
def frames_and_literature_positions_to_atom14_pos(r: Rigid, aatype: torch.Tensor, default_frames: torch.Tensor, group_idx: torch.Tensor, atom_mask: torch.Tensor, lit_positions: torch.Tensor) -> torch.Tensor:
    group_mask = group_idx[aatype, ...]
    group_mask_one_hot: torch.LongTensor = nn.functional.one_hot(group_mask, num_classes=default_frames.shape[-3])
    t_atoms_to_global = r[..., None, :] * group_mask_one_hot
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(lambda x: torch.sum(x, dim=-1))
    atom_mask = atom_mask[aatype, ...].unsqueeze(-1)
    lit_positions = lit_positions[aatype, ...]
    pred_positions = t_atoms_to_global.apply(lit_positions)
    pred_positions = pred_positions * atom_mask
    return pred_positions