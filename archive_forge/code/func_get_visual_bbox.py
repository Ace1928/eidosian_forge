import collections
import logging
import math
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from transformers import UdopConfig
from transformers.modeling_outputs import (
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from ..deprecated._archive_maps import UDOP_PRETRAINED_MODEL_ARCHIVE_LIST  # noqa: F401, E402
def get_visual_bbox(image_size=224, patch_size=16):
    image_feature_pool_shape = [image_size // patch_size, image_size // patch_size]
    visual_bbox_x = torch.arange(0, 1.0 * (image_feature_pool_shape[1] + 1), 1.0)
    visual_bbox_x /= image_feature_pool_shape[1]
    visual_bbox_y = torch.arange(0, 1.0 * (image_feature_pool_shape[0] + 1), 1.0)
    visual_bbox_y /= image_feature_pool_shape[0]
    visual_bbox_input = torch.stack([visual_bbox_x[:-1].repeat(image_feature_pool_shape[0], 1), visual_bbox_y[:-1].repeat(image_feature_pool_shape[1], 1).transpose(0, 1), visual_bbox_x[1:].repeat(image_feature_pool_shape[0], 1), visual_bbox_y[1:].repeat(image_feature_pool_shape[1], 1).transpose(0, 1)], dim=-1)
    visual_bbox_input = visual_bbox_input.view(-1, 4)
    return visual_bbox_input