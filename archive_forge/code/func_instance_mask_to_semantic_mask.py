from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from tqdm.auto import tqdm
from ultralytics.engine.results import Results
from ultralytics.models.yolo.segment import SegmentationPredictor
from ultralytics.utils.ops import scale_image
import wandb
from wandb.integration.ultralytics.bbox_utils import (
def instance_mask_to_semantic_mask(instance_mask, class_indices):
    height, width, num_instances = instance_mask.shape
    semantic_mask = np.zeros((height, width), dtype=np.uint8)
    for i in range(num_instances):
        instance_map = instance_mask[:, :, i]
        class_index = class_indices[i]
        semantic_mask[instance_map == 1] = class_index
    return semantic_mask