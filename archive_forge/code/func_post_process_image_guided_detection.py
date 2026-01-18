import warnings
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import (
from ...image_utils import (
from ...utils import (
def post_process_image_guided_detection(self, outputs, threshold=0.0, nms_threshold=0.3, target_sizes=None):
    """
        Converts the output of [`OwlViTForObjectDetection.image_guided_detection`] into the format expected by the COCO
        api.

        Args:
            outputs ([`OwlViTImageGuidedObjectDetectionOutput`]):
                Raw outputs of the model.
            threshold (`float`, *optional*, defaults to 0.0):
                Minimum confidence threshold to use to filter out predicted boxes.
            nms_threshold (`float`, *optional*, defaults to 0.3):
                IoU threshold for non-maximum suppression of overlapping boxes.
            target_sizes (`torch.Tensor`, *optional*):
                Tensor of shape (batch_size, 2) where each entry is the (height, width) of the corresponding image in
                the batch. If set, predicted normalized bounding boxes are rescaled to the target sizes. If left to
                None, predictions will not be unnormalized.

        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model. All labels are set to None as
            `OwlViTForObjectDetection.image_guided_detection` perform one-shot object detection.
        """
    logits, target_boxes = (outputs.logits, outputs.target_pred_boxes)
    if len(logits) != len(target_sizes):
        raise ValueError('Make sure that you pass in as many target sizes as the batch dimension of the logits')
    if target_sizes.shape[1] != 2:
        raise ValueError('Each element of target_sizes must contain the size (h, w) of each image of the batch')
    probs = torch.max(logits, dim=-1)
    scores = torch.sigmoid(probs.values)
    target_boxes = center_to_corners_format(target_boxes)
    if nms_threshold < 1.0:
        for idx in range(target_boxes.shape[0]):
            for i in torch.argsort(-scores[idx]):
                if not scores[idx][i]:
                    continue
                ious = box_iou(target_boxes[idx][i, :].unsqueeze(0), target_boxes[idx])[0][0]
                ious[i] = -1.0
                scores[idx][ious > nms_threshold] = 0.0
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(target_boxes.device)
    target_boxes = target_boxes * scale_fct[:, None, :]
    results = []
    alphas = torch.zeros_like(scores)
    for idx in range(target_boxes.shape[0]):
        query_scores = scores[idx]
        if not query_scores.nonzero().numel():
            continue
        query_scores[query_scores < threshold] = 0.0
        max_score = torch.max(query_scores) + 1e-06
        query_alphas = (query_scores - max_score * 0.1) / (max_score * 0.9)
        query_alphas = torch.clip(query_alphas, 0.0, 1.0)
        alphas[idx] = query_alphas
        mask = alphas[idx] > 0
        box_scores = alphas[idx][mask]
        boxes = target_boxes[idx][mask]
        results.append({'scores': box_scores, 'labels': None, 'boxes': boxes})
    return results