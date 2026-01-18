import functools
import math
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.nn.functional import conv2d, conv3d, pad, unfold
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.imports import _SCIPY_AVAILABLE
def mask_edges(preds: Tensor, target: Tensor, crop: bool=True, spacing: Optional[Union[Tuple[int, int], Tuple[int, int, int]]]=None) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]:
    """Get the edges of binary segmentation masks.

    Args:
        preds: The predicted binary segmentation mask
        target: The ground truth binary segmentation mask
        crop: Whether to crop the edges to the region of interest. If ``True``, the edges are cropped to the bounding
        spacing: The pixel spacing of the input images. If provided, the edges are calculated using the euclidean

    Returns:
        If spacing is not provided, a 2-tuple containing the edges of the predicted and target mask respectively is
        returned. If spacing is provided, a 4-tuple containing the edges and areas of the predicted and target mask
        respectively is returned.

    """
    _check_same_shape(preds, target)
    if preds.ndim not in [2, 3]:
        raise ValueError(f'Expected argument `preds` to be of rank 2 or 3 but got rank `{preds.ndim}`.')
    check_if_binarized(preds)
    check_if_binarized(target)
    if crop:
        or_val = preds | target
        if not or_val.any():
            p, t = (torch.zeros_like(preds), torch.zeros_like(target))
            return (p, t, p, t)
        preds, target = (pad(preds, preds.ndim * [1, 1]), pad(target, target.ndim * [1, 1]))
    if spacing is None:
        be_pred = binary_erosion(preds.unsqueeze(0).unsqueeze(0)).squeeze() ^ preds
        be_target = binary_erosion(target.unsqueeze(0).unsqueeze(0)).squeeze() ^ target
        return (be_pred, be_target)
    table, kernel = get_neighbour_tables(spacing, device=preds.device)
    spatial_dims = len(spacing)
    conv_operator = conv2d if spatial_dims == 2 else conv3d
    volume = torch.stack([preds.unsqueeze(0), target.unsqueeze(0)], dim=0).float()
    code_preds, code_target = conv_operator(volume, kernel.to(volume))
    all_ones = len(table) - 1
    edges_preds = (code_preds != 0) & (code_preds != all_ones)
    edges_target = (code_target != 0) & (code_target != all_ones)
    areas_preds = torch.index_select(table, 0, code_preds.view(-1).int()).view_as(code_preds)
    areas_target = torch.index_select(table, 0, code_target.view(-1).int()).view_as(code_target)
    return (edges_preds[0], edges_target[0], areas_preds[0], areas_target[0])