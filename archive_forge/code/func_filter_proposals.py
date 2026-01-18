from typing import Dict, List, Optional, Tuple
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops, Conv2dNormActivation
from . import _utils as det_utils
from .anchor_utils import AnchorGenerator  # noqa: 401
from .image_list import ImageList
def filter_proposals(self, proposals: Tensor, objectness: Tensor, image_shapes: List[Tuple[int, int]], num_anchors_per_level: List[int]) -> Tuple[List[Tensor], List[Tensor]]:
    num_images = proposals.shape[0]
    device = proposals.device
    objectness = objectness.detach()
    objectness = objectness.reshape(num_images, -1)
    levels = [torch.full((n,), idx, dtype=torch.int64, device=device) for idx, n in enumerate(num_anchors_per_level)]
    levels = torch.cat(levels, 0)
    levels = levels.reshape(1, -1).expand_as(objectness)
    top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)
    image_range = torch.arange(num_images, device=device)
    batch_idx = image_range[:, None]
    objectness = objectness[batch_idx, top_n_idx]
    levels = levels[batch_idx, top_n_idx]
    proposals = proposals[batch_idx, top_n_idx]
    objectness_prob = torch.sigmoid(objectness)
    final_boxes = []
    final_scores = []
    for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
        boxes = box_ops.clip_boxes_to_image(boxes, img_shape)
        keep = box_ops.remove_small_boxes(boxes, self.min_size)
        boxes, scores, lvl = (boxes[keep], scores[keep], lvl[keep])
        keep = torch.where(scores >= self.score_thresh)[0]
        boxes, scores, lvl = (boxes[keep], scores[keep], lvl[keep])
        keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)
        keep = keep[:self.post_nms_top_n()]
        boxes, scores = (boxes[keep], scores[keep])
        final_boxes.append(boxes)
        final_scores.append(scores)
    return (final_boxes, final_scores)