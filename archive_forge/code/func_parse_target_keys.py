from __future__ import annotations
import collections.abc
import contextlib
from collections import defaultdict
from copy import copy
import torch
from torchvision import datasets, tv_tensors
from torchvision.transforms.v2 import functional as F
def parse_target_keys(target_keys, *, available, default):
    if target_keys is None:
        target_keys = default
    if target_keys == 'all':
        target_keys = available
    else:
        target_keys = set(target_keys)
        extra = target_keys - available
        if extra:
            raise ValueError(f'Target keys {sorted(extra)} are not available')
    return target_keys