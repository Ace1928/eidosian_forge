import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_funnel import FunnelConfig
def init_attention_inputs(self, inputs_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor]:
    """Returns the attention inputs associated to the inputs of the model."""
    self.pooling_mult = 1
    self.seq_len = seq_len = inputs_embeds.size(1)
    position_embeds = self.get_position_embeds(seq_len, inputs_embeds.dtype, inputs_embeds.device)
    token_type_mat = self.token_type_ids_to_mat(token_type_ids) if token_type_ids is not None else None
    cls_mask = nn.functional.pad(inputs_embeds.new_ones([seq_len - 1, seq_len - 1]), (1, 0, 1, 0)) if self.config.separate_cls else None
    return (position_embeds, token_type_mat, attention_mask, cls_mask)