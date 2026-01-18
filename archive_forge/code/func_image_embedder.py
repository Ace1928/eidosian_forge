import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_owlv2 import Owlv2Config, Owlv2TextConfig, Owlv2VisionConfig
def image_embedder(self, pixel_values: torch.FloatTensor, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None) -> Tuple[torch.FloatTensor]:
    vision_outputs = self.owlv2.vision_model(pixel_values=pixel_values, return_dict=True)
    last_hidden_state = vision_outputs[0]
    image_embeds = self.owlv2.vision_model.post_layernorm(last_hidden_state)
    new_size = tuple(np.array(image_embeds.shape) - np.array((0, 1, 0)))
    class_token_out = torch.broadcast_to(image_embeds[:, :1, :], new_size)
    image_embeds = image_embeds[:, 1:, :] * class_token_out
    image_embeds = self.layer_norm(image_embeds)
    new_size = (image_embeds.shape[0], int(np.sqrt(image_embeds.shape[1])), int(np.sqrt(image_embeds.shape[1])), image_embeds.shape[-1])
    image_embeds = image_embeds.reshape(new_size)
    return (image_embeds, vision_outputs)