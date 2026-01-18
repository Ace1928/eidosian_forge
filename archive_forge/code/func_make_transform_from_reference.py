from __future__ import annotations
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
@staticmethod
def make_transform_from_reference(n_xyz: torch.Tensor, ca_xyz: torch.Tensor, c_xyz: torch.Tensor, eps: float=1e-20) -> Rigid:
    """
        Returns a transformation object from reference coordinates.

        Note that this method does not take care of symmetries. If you provide the atom positions in the non-standard
        way, the N atom will end up not at [-0.527250, 1.359329, 0.0] but instead at [-0.527250, -1.359329, 0.0]. You
        need to take care of such cases in your code.

        Args:
            n_xyz: A [*, 3] tensor of nitrogen xyz coordinates.
            ca_xyz: A [*, 3] tensor of carbon alpha xyz coordinates.
            c_xyz: A [*, 3] tensor of carbon xyz coordinates.
        Returns:
            A transformation object. After applying the translation and rotation to the reference backbone, the
            coordinates will approximately equal to the input coordinates.
        """
    translation = -1 * ca_xyz
    n_xyz = n_xyz + translation
    c_xyz = c_xyz + translation
    c_x, c_y, c_z = [c_xyz[..., i] for i in range(3)]
    norm = torch.sqrt(eps + c_x ** 2 + c_y ** 2)
    sin_c1 = -c_y / norm
    cos_c1 = c_x / norm
    c1_rots = sin_c1.new_zeros((*sin_c1.shape, 3, 3))
    c1_rots[..., 0, 0] = cos_c1
    c1_rots[..., 0, 1] = -1 * sin_c1
    c1_rots[..., 1, 0] = sin_c1
    c1_rots[..., 1, 1] = cos_c1
    c1_rots[..., 2, 2] = 1
    norm = torch.sqrt(eps + c_x ** 2 + c_y ** 2 + c_z ** 2)
    sin_c2 = c_z / norm
    cos_c2 = torch.sqrt(c_x ** 2 + c_y ** 2) / norm
    c2_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
    c2_rots[..., 0, 0] = cos_c2
    c2_rots[..., 0, 2] = sin_c2
    c2_rots[..., 1, 1] = 1
    c2_rots[..., 2, 0] = -1 * sin_c2
    c2_rots[..., 2, 2] = cos_c2
    c_rots = rot_matmul(c2_rots, c1_rots)
    n_xyz = rot_vec_mul(c_rots, n_xyz)
    _, n_y, n_z = [n_xyz[..., i] for i in range(3)]
    norm = torch.sqrt(eps + n_y ** 2 + n_z ** 2)
    sin_n = -n_z / norm
    cos_n = n_y / norm
    n_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
    n_rots[..., 0, 0] = 1
    n_rots[..., 1, 1] = cos_n
    n_rots[..., 1, 2] = -1 * sin_n
    n_rots[..., 2, 1] = sin_n
    n_rots[..., 2, 2] = cos_n
    rots = rot_matmul(n_rots, c_rots)
    rots = rots.transpose(-1, -2)
    translation = -1 * translation
    rot_obj = Rotation(rot_mats=rots, quats=None)
    return Rigid(rot_obj, translation)