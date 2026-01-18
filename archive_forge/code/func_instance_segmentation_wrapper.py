from __future__ import annotations
import collections.abc
import contextlib
from collections import defaultdict
from copy import copy
import torch
from torchvision import datasets, tv_tensors
from torchvision.transforms.v2 import functional as F
def instance_segmentation_wrapper(mask):
    data = pil_image_to_mask(mask)
    masks = []
    labels = []
    for id in data.unique():
        masks.append(data == id)
        label = id
        if label >= 1000:
            label //= 1000
        labels.append(label)
    return dict(masks=tv_tensors.Mask(torch.stack(masks)), labels=torch.stack(labels))