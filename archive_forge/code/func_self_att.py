import math
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, SmoothL1Loss
from ...activations import ACT2FN, gelu
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_lxmert import LxmertConfig
def self_att(self, lang_input, lang_attention_mask, visual_input, visual_attention_mask):
    lang_att_output = self.lang_self_att(lang_input, lang_attention_mask, output_attentions=False)
    visual_att_output = self.visn_self_att(visual_input, visual_attention_mask, output_attentions=False)
    return (lang_att_output[0], visual_att_output[0])