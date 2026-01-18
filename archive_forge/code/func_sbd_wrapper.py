from __future__ import annotations
import collections.abc
import contextlib
from collections import defaultdict
from copy import copy
import torch
from torchvision import datasets, tv_tensors
from torchvision.transforms.v2 import functional as F
@WRAPPER_FACTORIES.register(datasets.SBDataset)
def sbd_wrapper(dataset, target_keys):
    if dataset.mode == 'boundaries':
        raise_not_supported("SBDataset with mode='boundaries'")
    return segmentation_wrapper_factory(dataset, target_keys)