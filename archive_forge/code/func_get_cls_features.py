import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN, QuickGELUActivation
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel, apply_chunking_to_forward
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_bridgetower import BridgeTowerConfig, BridgeTowerTextConfig, BridgeTowerVisionConfig
def get_cls_features(self, text_features, image_features):
    cls_features_text = self.cross_modal_text_pooler(text_features)
    cls_features_image = self.cross_modal_image_pooler(image_features)
    return torch.cat([cls_features_text, cls_features_image], dim=-1)