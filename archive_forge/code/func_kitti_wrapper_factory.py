from __future__ import annotations
import collections.abc
import contextlib
from collections import defaultdict
from copy import copy
import torch
from torchvision import datasets, tv_tensors
from torchvision.transforms.v2 import functional as F
@WRAPPER_FACTORIES.register(datasets.Kitti)
def kitti_wrapper_factory(dataset, target_keys):
    target_keys = parse_target_keys(target_keys, available={'type', 'truncated', 'occluded', 'alpha', 'bbox', 'dimensions', 'location', 'rotation_y', 'boxes', 'labels'}, default={'boxes', 'labels'})

    def wrapper(idx, sample):
        image, target = sample
        if target is None:
            return (image, target)
        batched_target = list_of_dicts_to_dict_of_lists(target)
        target = {}
        if 'boxes' in target_keys:
            target['boxes'] = tv_tensors.BoundingBoxes(batched_target['bbox'], format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(image.height, image.width))
        if 'labels' in target_keys:
            target['labels'] = torch.tensor([KITTI_CATEGORY_TO_IDX[category] for category in batched_target['type']])
        for target_key in target_keys - {'boxes', 'labels'}:
            target[target_key] = batched_target[target_key]
        return (image, target)
    return wrapper