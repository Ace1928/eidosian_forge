from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from tqdm.auto import tqdm
from ultralytics.engine.results import Results
from ultralytics.models.yolo.segment import SegmentationPredictor
from ultralytics.utils.ops import scale_image
import wandb
from wandb.integration.ultralytics.bbox_utils import (
def get_boxes_and_masks(result: Results) -> Tuple[Dict, Dict, Dict]:
    boxes = result.boxes.xywh.long().numpy()
    classes = result.boxes.cls.long().numpy()
    confidence = result.boxes.conf.numpy()
    class_id_to_label = {int(k): str(v) for k, v in result.names.items()}
    class_id_to_label.update({len(result.names.items()): 'background'})
    mean_confidence_map = get_mean_confidence_map(classes, confidence, class_id_to_label)
    masks = None
    if result.masks is not None:
        scaled_instance_mask = scale_image(np.transpose(result.masks.data.numpy(), (1, 2, 0)), result.orig_img[:, :, ::-1].shape)
        scaled_semantic_mask = instance_mask_to_semantic_mask(scaled_instance_mask, classes.tolist())
        scaled_semantic_mask[scaled_semantic_mask == 0] = len(result.names.items())
        masks = {'predictions': {'mask_data': scaled_semantic_mask, 'class_labels': class_id_to_label}}
    box_data, total_confidence = ([], 0.0)
    for idx in range(len(boxes)):
        box_data.append({'position': {'middle': [int(boxes[idx][0]), int(boxes[idx][1])], 'width': int(boxes[idx][2]), 'height': int(boxes[idx][3])}, 'domain': 'pixel', 'class_id': int(classes[idx]), 'box_caption': class_id_to_label[int(classes[idx])], 'scores': {'confidence': float(confidence[idx])}})
        total_confidence += float(confidence[idx])
    boxes = {'predictions': {'box_data': box_data, 'class_labels': class_id_to_label}}
    return (boxes, masks, mean_confidence_map)