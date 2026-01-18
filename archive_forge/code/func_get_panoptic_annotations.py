import json
import os
import warnings
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
from ...image_utils import (
from ...utils import (
def get_panoptic_annotations(self, label, num_class_obj):
    annotation_classes = label['classes']
    annotation_masks = label['masks']
    texts = ['an panoptic photo'] * self.num_text
    classes = []
    masks = []
    for idx in range(len(annotation_classes)):
        class_id = annotation_classes[idx]
        mask = annotation_masks[idx].data
        if not np.all(mask is False):
            cls_name = self.metadata[str(class_id)]
            classes.append(class_id)
            masks.append(mask)
            num_class_obj[cls_name] += 1
    num = 0
    for i, cls_name in enumerate(self.metadata['class_names']):
        if num_class_obj[cls_name] > 0:
            for _ in range(num_class_obj[cls_name]):
                if num >= len(texts):
                    break
                texts[num] = f'a photo with a {cls_name}'
                num += 1
    classes = np.array(classes)
    masks = np.array(masks)
    return (classes, masks, texts)