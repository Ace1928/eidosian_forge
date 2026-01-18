import copy
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_oneformer import OneFormerConfig
def loss_masks(self, masks_queries_logits: Tensor, mask_labels: List[Tensor], indices: Tuple[np.array], num_masks: int) -> Dict[str, Tensor]:
    """Compute the losses related to the masks using focal and dice loss.

        Args:
            masks_queries_logits (`torch.Tensor`):
                A tensor of shape `batch_size, num_queries, height, width`
            mask_labels (`torch.Tensor`):
                List of mask labels of shape `(labels, height, width)`.
            indices (`Tuple[np.array])`:
                The indices computed by the Hungarian matcher.
            num_masks (`int)`:
                The number of masks, used for normalization.

        Returns:
            `Dict[str, Tensor]`: A dict of `torch.Tensor` containing two keys:
            - **loss_mask** -- The loss computed using sigmoid ce loss on the predicted and ground truth masks.
            - **loss_dice** -- The loss computed using dice loss on the predicted on the predicted and ground truth
              masks.
        """
    src_idx = self._get_predictions_permutation_indices(indices)
    tgt_idx = self._get_targets_permutation_indices(indices)
    pred_masks = masks_queries_logits[src_idx]
    target_masks, _ = self._pad_images_to_max_in_batch(mask_labels)
    target_masks = target_masks[tgt_idx]
    pred_masks = pred_masks[:, None]
    target_masks = target_masks[:, None]
    with torch.no_grad():
        point_coords = self.sample_points_using_uncertainty(pred_masks, self.calculate_uncertainty, self.num_points, self.oversample_ratio, self.importance_sample_ratio)
        point_labels = sample_point(target_masks, point_coords, align_corners=False).squeeze(1)
    point_logits = sample_point(pred_masks, point_coords, align_corners=False).squeeze(1)
    losses = {'loss_mask': sigmoid_cross_entropy_loss(point_logits, point_labels, num_masks), 'loss_dice': dice_loss(point_logits, point_labels, num_masks)}
    del pred_masks
    del target_masks
    return losses