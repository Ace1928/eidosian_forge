from __future__ import annotations
import collections.abc
import contextlib
from collections import defaultdict
from copy import copy
import torch
from torchvision import datasets, tv_tensors
from torchvision.transforms.v2 import functional as F
def classification_wrapper_factory(dataset, target_keys):
    return identity_wrapper_factory(dataset, target_keys)