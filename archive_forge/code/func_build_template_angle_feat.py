from typing import Dict, Tuple, overload
import torch
import torch.types
from torch import nn
from . import residue_constants as rc
from .rigid_utils import Rigid, Rotation
from .tensor_utils import batched_gather
def build_template_angle_feat(template_feats: Dict[str, torch.Tensor]) -> torch.Tensor:
    template_aatype = template_feats['template_aatype']
    torsion_angles_sin_cos = template_feats['template_torsion_angles_sin_cos']
    alt_torsion_angles_sin_cos = template_feats['template_alt_torsion_angles_sin_cos']
    torsion_angles_mask = template_feats['template_torsion_angles_mask']
    template_angle_feat = torch.cat([nn.functional.one_hot(template_aatype, 22), torsion_angles_sin_cos.reshape(*torsion_angles_sin_cos.shape[:-2], 14), alt_torsion_angles_sin_cos.reshape(*alt_torsion_angles_sin_cos.shape[:-2], 14), torsion_angles_mask], dim=-1)
    return template_angle_feat