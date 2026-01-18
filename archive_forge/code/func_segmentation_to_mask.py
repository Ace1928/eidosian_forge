from __future__ import annotations
import collections.abc
import contextlib
from collections import defaultdict
from copy import copy
import torch
from torchvision import datasets, tv_tensors
from torchvision.transforms.v2 import functional as F
def segmentation_to_mask(segmentation, *, canvas_size):
    from pycocotools import mask
    segmentation = mask.frPyObjects(segmentation, *canvas_size) if isinstance(segmentation, dict) else mask.merge(mask.frPyObjects(segmentation, *canvas_size))
    return torch.from_numpy(mask.decode(segmentation))