import copy
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_oneformer import OneFormerConfig
def forward_post(self, output, memory, output_mask: Optional[Tensor]=None, memory_mask: Optional[Tensor]=None, output_key_padding_mask: Optional[Tensor]=None, memory_key_padding_mask: Optional[Tensor]=None, pos: Optional[Tensor]=None, query_pos: Optional[Tensor]=None):
    q = k = self.with_pos_embed(output, query_pos)
    output2 = self.self_attn(q, k, value=output, attn_mask=output_mask, key_padding_mask=output_key_padding_mask)
    output2 = output2[0]
    output = output + self.dropout1(output2)
    output = self.norm1(output)
    output2 = self.multihead_attn(query=self.with_pos_embed(output, query_pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
    output2 = output2[0]
    output = output + self.dropout2(output2)
    output = self.norm2(output)
    output2 = self.linear2(self.dropout(self.activation(self.linear1(output))))
    output = output + self.dropout3(output2)
    output = self.norm3(output)
    return output