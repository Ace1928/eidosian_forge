import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_realm import RealmConfig
def block_embedding_to(self, device):
    """Send `self.block_emb` to a specific device.

        Args:
            device (`str` or `torch.device`):
                The device to which `self.block_emb` will be sent.
        """
    self.block_emb = self.block_emb.to(device)