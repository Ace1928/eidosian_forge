import collections
import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_layoutlmv3 import LayoutLMv3Config
def calculate_visual_bbox(self, device, dtype, batch_size):
    visual_bbox = self.visual_bbox.repeat(batch_size, 1, 1)
    visual_bbox = visual_bbox.to(device).type(dtype)
    return visual_bbox