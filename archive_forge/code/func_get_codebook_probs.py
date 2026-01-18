import collections
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_flava import (
def get_codebook_probs(self, pixel_values: torch.Tensor) -> torch.Tensor:
    z_logits = self.blocks(pixel_values)
    return nn.Softmax(dim=1)(z_logits)