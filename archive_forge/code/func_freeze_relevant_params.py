from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ... import PreTrainedModel
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PretrainedConfig
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_idefics import IdeficsConfig
from .perceiver import IdeficsPerceiverResampler
from .vision import IdeficsVisionTransformer
def freeze_relevant_params(self, config=None):
    if config is None:
        config = self.config
    if config.freeze_text_layers:
        self.freeze_text_layers(config.freeze_text_module_exceptions)
    if config.freeze_vision_layers:
        freeze_model(self.vision_model, module_exceptions=config.freeze_vision_module_exceptions)