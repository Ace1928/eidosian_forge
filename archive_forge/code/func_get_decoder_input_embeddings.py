import copy
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_marian import MarianConfig
def get_decoder_input_embeddings(self):
    if self.config.share_encoder_decoder_embeddings:
        raise ValueError('`get_decoder_input_embeddings` should not be called if `config.share_encoder_decoder_embeddings` is `True`. Please use `get_input_embeddings` instead.')
    return self.get_decoder().get_input_embeddings()