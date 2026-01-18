import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_kosmos2 import Kosmos2Config, Kosmos2TextConfig, Kosmos2VisionConfig
def forward_embedding(self, input_ids, inputs_embeds: torch.Tensor=None, image_embeds: torch.Tensor=None, img_input_mask: torch.Tensor=None, past_key_values_length: int=0, position_ids: torch.Tensor=None):
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    if image_embeds is not None:
        inputs_embeds[img_input_mask.to(dtype=torch.bool)] = image_embeds.to(inputs_embeds.device).view(-1, image_embeds.size(-1))
    inputs_embeds = inputs_embeds * self.embed_scale
    positions = self.embed_positions(input_ids=input_ids, inputs_embeds=inputs_embeds, past_key_values_length=past_key_values_length, position_ids=position_ids)
    positions = positions.to(inputs_embeds.device)
    hidden_states = inputs_embeds + positions
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    return hidden_states